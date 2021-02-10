#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : multi_simple.py
# @Author: sl
# @Date  : 2021/2/10 -  下午9:17


"""
多层感知机 简单实现

"""

import torch
from time import time
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
import torch.utils.data as Data
from torch import nn
from torch.nn import init
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

sys.path.append("..")
import deep.torch.d2lzh_pytorch as d2l


class MultiLinearNet(nn.Module):

    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super(MultiLinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_hiddens)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_hiddens, num_outputs)

    def forward(self, x):  # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        y = self.relu(y)
        y = self.linear2(y)
        return y

        return y



if __name__ == '__main__':

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 28 * 28
    num_inputs = 784
    num_outputs = 10
    num_hiddens = 256

    net = MultiLinearNet(num_inputs, num_outputs, num_hiddens)

    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)

    loss = nn.CrossEntropyLoss()
    lr = 0.5
    optimizer = torch.optim.SGD(net.parameters(), lr)

    num_epochs = 10
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

    X, y = iter(test_iter).next()
    true_labels = d2l.get_fashion_mnish_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnish_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

    d2l.show_fashion_mnist(X[0:9], titles[0:9])

    pass
