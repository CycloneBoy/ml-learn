#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : linear_simple.py
# @Author: sl
# @Date  : 2021/2/10 -  上午10:56


"""
  线性回归: 简洁实现

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


sys.path.append("..")
import deep.torch.d2lzh_pytorch as d2l


class LinearNet(torch.nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


def test_net():
    """ 多种方式定义网络 """
    net = nn.Sequential(
        nn.Linear(num_inputs,1)
    )
    print(net)

    net = nn.Sequential()
    net.add_module('linear',nn.Linear(num_inputs,1))
    print(net)

    from collections import OrderedDict
    net = nn.Sequential(OrderedDict([
        ('linear',nn.Linear(num_inputs,1))
    ]))

    print(net)
    print(net[0])


def test_optimizer():
    """ 测试优化器 """
    optimizer = optim.SGD([
        # {'params':net.linear.parameters()},
        {'params': net.linear.parameters(), 'lr': 0.01}
    ], lr=0.03)
    print(optimizer)
    # 调整学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
    print(optimizer)


if __name__ == '__main__':

    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features, labels = d2l.prepare_linear_data(num_examples, num_inputs, true_w, true_b)

    print(features[0], labels[0])

    batch_size = 10
    # 将训练数据的特征和标签组合
    dataset = Data.TensorDataset(features, labels)
    # 随机选取小批量
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    for X, y in data_iter:
        print(X, y)
        break

    net = LinearNet(num_inputs)
    print(net)

    test_net()

    for param in net.parameters():
        print(param)

    # init.normal_(net[0].weight,mean=0,std=0.01)
    # init.constant_(net[0].bias,val=0)

    init.normal_(net.linear.weight,mean=0,std=0.01)
    init.constant_(net.linear.bias,val=0)

    for param in net.parameters():
        print(param)

    loss = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(),lr=0.03)
    print(optimizer)

    # test_optimizer()

    num_epochs = 30
    for epoch in range(1,num_epochs + 1):
        for X,y in data_iter:
            output = net(X)
            l = loss(output,y.view(-1,1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        print('epoch %d , loss: %f' %(epoch,l.item()))


    dense = net.linear
    print(true_w, '\n', dense.weight)
    print(true_b, '\n', dense.bias)

    pass
