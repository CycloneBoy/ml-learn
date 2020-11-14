#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_sentiment_classfication_cnn.py
# @Author: sl
# @Date  : 2020/10/18 - 下午3:04


import os
import sys

import torch
import torch.utils.data as Data
from torch import nn

from util.common_utils import get_glove
from util.constant import DATA_CACHE_DIR

sys.path.append("..")
import deep.torch.d2lzh_pytorch as d2l

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ROOT = DATA_CACHE_DIR


def corr1d(X, K):
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i:i + w] * K).sum()
    return Y


X, K = torch.tensor([0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2])
res = corr1d(X, K)
print(res)


def corr1d_multi_in(X, K):
    return torch.stack([corr1d(x, k) for x, k in zip(X, K)]).sum(dim=0)


X = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
                  [1, 2, 3, 4, 5, 6, 7],
                  [2, 3, 4, 5, 6, 7, 8]])
K = torch.tensor([[1, 2], [3, 4], [-1, -3]])
res = corr1d_multi_in(X, K)
print(res)

train_data, test_data = d2l.read_imdb('train'), d2l.read_imdb('test')
print("train_data size:", len(train_data))
print("test_data size:", len(test_data))

vocab = d2l.get_vocab_imdb(train_data)
print('words in vocab:', len(vocab))
# words in vocab: 46152

batch_size = 64
train_set = Data.TensorDataset(*d2l.preprocess_imdb(train_data, vocab))
test_set = Data.TensorDataset(*d2l.preprocess_imdb(test_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)

embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
net = d2l.TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
print(net)

glove_vocab = get_glove()

net.embedding.weight.url.copy_(
    d2l.load_pretrained_embedding(vocab.itos, glove_vocab))
net.constant_embedding.weight.url.copy_(
    d2l.load_pretrained_embedding(vocab.itos, glove_vocab))
net.constant_embedding.weight.requires_grad = False  # 直接加载预训练好的, 所以不需要更新它

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

res = d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])  # positive
print(res)

res = d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])  # negative
print(res)

if __name__ == '__main__':
    pass
