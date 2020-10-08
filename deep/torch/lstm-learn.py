#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : lstm-learn.py
# @Author: sl
# @Date  : 2020/10/7 - 下午10:43

import sys

sys.path.append("..")

import numpy as np

import torch
from torch import nn

import deep.torch.d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from util.logger_utils import get_log

log = get_log("{}.log".format("lstm-learn"))

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()


def test_one_hot():
    x = torch.tensor([0, 2])
    return d2l.one_hot(x, vocab_size)


def test_to_onehot():
    X = torch.arange(10).view(2, 5)
    inputs = d2l.to_onehot(X, vocab_size)
    print(len(inputs), inputs[0].shape)


num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01,size=shape),device=device,dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


def test_rnn():
    X = torch.arange(10).view(2, 5)
    state = init_rnn_state(X.shape[0],num_hiddens,device)
    inputs = d2l.to_onehot(X.to(device),vocab_size)
    params = get_params()
    outputs,state_new = rnn(inputs,state,params)
    print(len(outputs), outputs[0].shape, state_new[0].shape)

    res = d2l.predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
                device, idx_to_char, char_to_idx)
    print(res)


if __name__ == '__main__':
    # res = read_jay()
    # log.info("{}".format(res))

    # res = index_jay()
    # log.info("{}".format(res))

    # res = test_one_hot()
    # log.info("{}".format(res))
    #
    # test_to_onehot()
    test_rnn()


    pass
