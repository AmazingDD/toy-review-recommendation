'''
Data pre process

@author:
Chong Chen (cstchenc@163.com)

@ created:
25/8/2017
@references:
'''
import os
import json
import pandas as pd
import pickle
import numpy as np
TPS_DIR = '../data/music'
TP_file = os.path.join(TPS_DIR, 'Digital_Music_5.json')

# f= open(TP_file)
# users_id=[]
# items_id=[]
# ratings=[]
# reviews=[]
# np.random.seed(2017)

# for line in f:
#     js=json.loads(line)
#     if str(js['reviewerID'])=='unknown':
#         print("unknown")
#         continue
#     if str(js['asin'])=='unknown':
#         print("unknown2")
#         continue
#     reviews.append(js['reviewText'])
#     users_id.append(str(js['reviewerID'])+',')
#     items_id.append(str(js['asin'])+',')
#     ratings.append(str(js['overall']))
# data=pd.DataFrame({'user_id':pd.Series(users_id),
#                    'item_id':pd.Series(items_id),
#                    'ratings':pd.Series(ratings),
#                    'reviews':pd.Series(reviews)})[['user_id','item_id','ratings','reviews']]
# data.sample(frac=0.5, replace=True, random_state=2019).reset_index(drop=True)

# tmp operation
train = pd.read_csv(os.path.join(TPS_DIR, 'train.csv'))[['raw_userid', 'raw_itemid', 'rating', 'n_content']]
valid = pd.read_csv(os.path.join(TPS_DIR, 'valid_foldin.csv'))[['raw_userid', 'raw_itemid', 'rating', 'n_content']]
test = pd.read_csv(os.path.join(TPS_DIR, 'test_foldin.csv'))[['raw_userid', 'raw_itemid', 'rating', 'n_content']]
train = train.dropna()
valid = valid.dropna()
test = test.dropna()
train_idx = len(train) - 1
valid_idx = train_idx + len(valid)
test_idx = valid_idx + len(test)
data = pd.concat([train, valid, test], ignore_index=True)
data.rename(columns={'raw_userid': 'user_id', 
                     'raw_itemid': 'item_id',
                     'rating': 'ratings', 
                     'n_content': 'reviews'}, inplace=True)
###############

def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count
usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')
unique_uid = usercount.index
unique_sid = itemcount.index
item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
def numerize(tp):
    uid = map(lambda x: user2id[x], tp['user_id'])
    sid = map(lambda x: item2id[x], tp['item_id'])
    tp['user_id'] = list(uid)
    tp['item_id'] = list(sid)
    return tp

data=numerize(data)
tp_rating=data[['user_id','item_id','ratings']]


# n_ratings = tp_rating.shape[0]
# test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
# test_idx = np.zeros(n_ratings, dtype=bool)
# test_idx[test] = True

# tp_1 = tp_rating[test_idx]
# tp_train= tp_rating[~test_idx]

# data2=data[test_idx]
# data=data[~test_idx]


# n_ratings = tp_1.shape[0]
# test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

# test_idx = np.zeros(n_ratings, dtype=bool)
# test_idx[test] = True

# tp_test = tp_1[test_idx]
# tp_valid = tp_1[~test_idx]
# tp_train.to_csv(os.path.join(TPS_DIR, 'music_train.csv'), index=False,header=None)
# tp_valid.to_csv(os.path.join(TPS_DIR, 'music_valid.csv'), index=False,header=None)
# tp_test.to_csv(os.path.join(TPS_DIR, 'music_test.csv'), index=False,header=None)

# tmp operation
tp_train = tp_rating.iloc[:train_idx, :]
tp_valid = tp_rating.iloc[train_idx:valid_idx, :]
tp_test = tp_rating.iloc[valid_idx:test_idx, :]
tp_train.to_csv(os.path.join(TPS_DIR, 'music_train.csv'), index=False,header=None)
tp_valid.to_csv(os.path.join(TPS_DIR, 'music_valid.csv'), index=False,header=None)
tp_test.to_csv(os.path.join(TPS_DIR, 'music_test.csv'), index=False,header=None)

data2 = data[valid_idx:test_idx]
data = data[:valid_idx]
###############

user_reviews={}
item_reviews={}
user_rid={}
item_rid={}
for i in data.values:
    if i[0] in user_reviews.keys():
        user_reviews[i[0]].append(i[3])
        user_rid[i[0]].append(i[1])
    else:
        user_rid[i[0]]=[i[1]]
        user_reviews[i[0]]=[i[3]]
    if i[1] in item_reviews.keys():
        item_reviews[i[1]].append(i[3])
        item_rid[i[1]].append(i[0])
    else:
        item_reviews[i[1]] = [i[3]]
        item_rid[i[1]]=[i[0]]


for i in data2.values:
    if i[0] in user_reviews.keys():
        l=1
    else:
        user_rid[i[0]]=[0]
        user_reviews[i[0]]=['0']
    if i[1] in item_reviews.keys():
        l=1
    else:
        item_reviews[i[1]] = [0]
        item_rid[i[1]]=['0']

print(item_reviews[11])
pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), 'wb'))
pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))

usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')


print(np.sort(np.array(usercount.values)))

print(np.sort(np.array(itemcount.values)))
