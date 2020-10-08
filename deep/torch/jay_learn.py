#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : jay_learn.py
# @Author: sl
# @Date  : 2020/10/8 - 下午9:32

import sys

sys.path.append("..")

import numpy as np
import time
import math

import torch
from torch import nn,optim
import torch.nn.functional as F

import deep.torch.d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from util.logger_utils import get_log

log = get_log("{}.log".format("jay-learn"))

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()


num_hiddens = 256

# rnn_layer = nn.LSTM(input_size=vocab_size,hidden_size=num_hiddens)
rnn_layer = nn.RNN(input_size=vocab_size,hidden_size=num_hiddens)

num_steps = 35
batch_size = 2
state = None

X = torch.rand(num_steps,batch_size,vocab_size)
Y,state_new = rnn_layer(X,state)
print(Y.shape,len(state_new),state_new[0].shape)

class RNNModel(nn.Module):

    def __init__(self,rnn_layer,vocab_size):
        super(RNNModel,self).__init__()
        self.rnn =rnn_layer
        self.hidden_size = rnn_layer.hidden_size * ( 2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense  = nn.Linear(self.hidden_size,vocab_size)
        self.state = None

    def forward(self, inputs, state): # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        X = d2l.to_onehot(inputs,self.vocab_size)# X是个list
        Y,self.state = self.rnn(torch.stack(X),state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1,Y.shape[-1]))
        return output,self.state


def test_torch_jay():
    model = RNNModel(rnn_layer,vocab_size).to(device)
    res = d2l.predict_rnn_pytorch('分开',10,model,vocab_size,device,idx_to_char,char_to_idx)
    print(res)

def test_gen():
    model = RNNModel(rnn_layer, vocab_size).to(device)
    num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2  # 注意这里的学习率设置
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes)


if __name__ == '__main__':
    # test_torch_jay()
    test_gen()
    pass