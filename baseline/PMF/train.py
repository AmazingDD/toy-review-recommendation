'''
@Author: Yu Di
@Date: 2020-01-10 22:15:56
@LastEditors  : Yudi
@LastEditTime : 2020-01-11 00:13:08
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.utils.data as data

from PMFRecommender import PointMF

def recall_at_k(rs, test_ur, k):
    assert k >= 1
    res = []
    for user in test_ur.keys():
        r = np.asarray(rs[user])[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        if len(test_ur[user]) == 0:
            raise KeyError(f'Invalid User Index: {user}')
        res.append(sum(r) / len(test_ur[user]))

    return np.mean(res)

def dcg_at_k(r, k):
    '''
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
    Returns:
        Discounted cumulative gain
    '''
    assert k >= 1
    r = np.asfarray(r)[:k] != 0
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    '''
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
    Returns:
        Normalized discounted cumulative gain
    '''
    assert k >= 1
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
                    type=str, 
                    default='music', 
                    help='select dataset')
parser.add_argument('--topk', 
                    type=int, 
                    default=50, 
                    help='top number of recommend list')
parser.add_argument('--loss_type', 
                    type=str, 
                    default='CL', 
                    help='loss function type')
parser.add_argument('--factors', 
                    type=int, 
                    default=32, 
                    help='predictive factors numbers in the model')
parser.add_argument('--epochs', 
                    type=int, 
                    default=50, 
                    help='training epochs')
parser.add_argument('--lr', 
                    type=float, 
                    default=0.01, 
                    help='learning rate')
parser.add_argument('--wd', 
                    type=float, 
                    default=0.001, 
                    help='model regularization rate')
parser.add_argument('--lamda', 
                    type=float, 
                    default=0.0, 
                    help='regularization weight')
parser.add_argument('--batch_size', 
                    type=int, 
                    default=512, 
                    help='batch size for training')
parser.add_argument('--gpu', 
                    type=str, 
                    default='0', 
                    help='gpu card ID')
args = parser.parse_args()

def get_ur(df):
    ur = defaultdict(set)
    for _, row in df.iterrows():
        ur[int(row['user'])].add(int(row['item']))

    return ur

class PointMFData(data.Dataset):
    def __init__(self, sampled_df):
        super(PointMFData, self).__init__()
        self.features_fill = []
        self.labels_fill = []
        for _, row in sampled_df.iterrows():
            self.features_fill.append([int(row['user']), int(row['item'])])
            self.labels_fill.append(row['rating'])
        self.labels_fill = np.array(self.labels_fill, dtype=np.float32)

    def __len__(self):
        return len(self.labels_fill)

    def __getitem__(self, idx):
        features = self.features_fill
        labels = self.labels_fill

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]

        return user, item, label

TPS_DIR = f'./data/{args.dataset}'
train = pd.read_csv(os.path.join(TPS_DIR, 'train.csv'))[['raw_userid', 'raw_itemid', 'rating', 'n_content']]
valid = pd.read_csv(os.path.join(TPS_DIR, 'valid_foldin.csv'))[['raw_userid', 'raw_itemid', 'rating', 'n_content']]
test = pd.read_csv(os.path.join(TPS_DIR, 'test_foldin.csv'))[['raw_userid', 'raw_itemid', 'rating', 'n_content']]
train = train.dropna()
valid = valid.dropna()
test = test.dropna()
train_idx = len(train) - 1
valid_idx = train_idx + len(valid)
test_idx = valid_idx + len(test)
df = pd.concat([train, valid, test], ignore_index=True)
df.rename(columns={'raw_userid': 'user', 
                     'raw_itemid': 'item',
                     'rating': 'rating', 
                     'n_content': 'reviews'}, inplace=True)
df.drop(['reviews'], axis=1, inplace=True)

df['user'] = pd.Categorical(df['user']).codes
df['item'] = pd.Categorical(df['item']).codes
df['rating'] = 1.0

train = df.iloc[:valid_idx, :]
test = df.iloc[valid_idx:, :]

user_num = df.user.nunique()
item_num = df.item.nunique()

test_ur = get_ur(test)
total_train_ur = get_ur(train)

item_pool = set(range(item_num))
print('='*50, '\n')

train_dataset = PointMFData(train)
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                               shuffle=True, num_workers=4)

model = PointMF(user_num, item_num, args.factors, args.lamda,
                args.epochs, args.lr, args.wd, args.gpu, args.loss_type)
model.fit(train_loader)

print('Start Calculating Metrics......')

test_ucands = defaultdict(list)
for k, v in test_ur.items():
    sub_item_pool = item_pool - total_train_ur[k] # remove GT & interacted
    test_ucands[k] = list(sub_item_pool)

print('')
print('Generate recommend list...')
print('')
preds = {}
for u in tqdm(test_ucands.keys()):
    # build a test MF dataset for certain user u
    tmp = pd.DataFrame({'user': [u for _ in test_ucands[u]], 
                        'item': test_ucands[u], 
                        'rating': [0. for _ in test_ucands[u]], # fake label, make nonsense
                        })
    tmp_dataset = PointMFData(tmp)
    tmp_loader = data.DataLoader(tmp_dataset, batch_size=len(tmp_dataset), 
                                 shuffle=False, num_workers=0)
    # get top-N list with torch method 
    for user_u, item_i, _ in tmp_loader:
        if torch.cuda.is_available():
            user_u = user_u.cuda()
            item_i = item_i.cuda()
        else:
            user_u = user_u.cpu()
            item_i = item_i.cpu()

        prediction = model.predict(user_u, item_i)
        _, indices = torch.topk(prediction, args.topk)
        top_n = torch.take(torch.tensor(test_ucands[u]), indices).cpu().numpy()
    preds[u] = top_n

# convert rank list to binary-interaction
for u in preds.keys():
    preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]

# process topN list and store result for reporting KPI
print('Save metric@k result to res folder...')
result_save_path = f'./res/{args.dataset}/'
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)

res = pd.DataFrame({'metric@K': ['rec', 'ndcg']})
for k in [1, 5, 10, 20, 30, 50]:
    if k > args.topk:
        continue
    tmp_preds = preds.copy()        
    tmp_preds = {key: rank_list[:k] for key, rank_list in tmp_preds.items()}

    rec_k = recall_at_k(tmp_preds, test_ur, k)
    ndcg_k = np.mean([ndcg_at_k(r, k) for r in tmp_preds.values()])

    if k == 10:
        print(f'Recall@{k}: {rec_k:.4f}')
        print(f'NDCG@{k}: {ndcg_k:.4f}')

    res[k] = np.array([rec_k, ndcg_k])

res.to_csv(f'{result_save_path}{args.prepro}_{args.test_method}_pointneumf_{args.loss_type}.csv', 
           index=False)
print('='* 20, ' Done ', '='*20)

