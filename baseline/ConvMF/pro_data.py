import os
import pandas as pd
import gzip
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_Digital_Music_5.json.gz')

df = df[['reviewerID', 'asin', 'overall', 'reviewText']].copy()

df['reviewText'] = df.reviewText.agg(lambda x: ' '.join(word_tokenize(x)))

f = open('music.txt', 'w+')
for _, row in df.iterrows():
    f.write(row[0] + '::' + row[1] + '::' + str(row[2]) + '::' + row[3] + '\n')
f.close()

if not os.path.exists('./pretrain/'):
    os.makedirs('./pretrain/')

# 裕训练word2vec
raw_sentences = df.reviewText.values.tolist()
sentences = [s.split() for s in raw_sentences]
model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
model.wv.save_word2vec_format('./pretrain/music_vectors.bin',binary=True)
