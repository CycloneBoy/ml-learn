#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_sentiment_classfication_cnn.py
# @Author: sl
# @Date  : 2020/10/18 - 下午3:04


import collections
import os
import random
import tarfile
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
import torch.nn.functional as F

from  tqdm import tqdm
import sys

from util.common_utils import get_glove
from util.constant import IMDB_DATA_DIR, DATA_CACHE_DIR

sys.path.append("..")
import deep.torch.d2lzh_pytorch as d2l


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ROOT = DATA_CACHE_DIR

def corr1d(X,K):
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i:i+w] * K).sum()
    return Y

X, K = torch.tensor([0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2])
res = corr1d(X, K)
print(res)

def corr1d_multi_in(X,K):
    return torch.stack([corr1d(x,k) for x,k in zip(X,K)]).sum(dim=0)


X = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = torch.tensor([[1, 2], [3, 4], [-1, -3]])
res = corr1d_multi_in(X, K)
print(res)

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
        return F.max_pool1d(x,kernel_size=x.shape[2])


train_data ,test_data = d2l.read_imdb('train'),d2l.read_imdb('test')
print("train_data size:",len(train_data))
print("test_data size:",len(test_data))

vocab = d2l.get_vocab_imdb(train_data)
print('words in vocab:', len(vocab))
# words in vocab: 46152

batch_size = 64
train_set = Data.TensorDataset(*d2l.preprocess_imdb(train_data, vocab))
test_set = Data.TensorDataset(*d2l.preprocess_imdb(test_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)


if __name__ == '__main__':
    pass