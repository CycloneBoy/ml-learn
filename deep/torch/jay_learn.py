#!/user/bin/env python
# -*- coding: utf-8 -*-
# rnn 实现
# @File  : jay_learn.py
# @Author: sl
# @Date  : 2020/10/8 - 下午9:32


import sys

sys.path.append("..")

import torch
from torch import nn

import deep.torch.d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from util.logger_utils import get_log

log = get_log("{}.log".format("jay-learn"))

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

num_hiddens = 256

# rnn_layer = nn.LSTM(input_size=vocab_size,hidden_size=num_hiddens)
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

num_steps = 35
batch_size = 2
state = None

X = torch.rand(num_steps, batch_size, vocab_size)
Y, state_new = rnn_layer(X, state)
print(Y.shape, len(state_new), state_new[0].shape)



def test_torch_jay():
    model = d2l.RNNModel(rnn_layer, vocab_size).to(device)
    res = d2l.predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)
    print(res)


def test_gen():
    model = d2l.RNNModel(rnn_layer, vocab_size).to(device)
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
