#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : softmax_simple.py
# @Author: sl
# @Date  : 2021/2/10 -  下午8:15

"""
softmax 回归 简单实现

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

class LinearNet(nn.Module):

    def __init__(self,num_inputs,num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs,num_outputs)

    def forward(self,x):# x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0],-1))
        return y

if __name__ == '__main__':
    batch_size = 256
    train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 28 * 28
    num_inputs = 784
    num_outputs = 10

    net = LinearNet(num_inputs,num_outputs)

    init.normal_(net.linear.weight,mean=0,std=0.01)
    init.constant_(net.linear.bias,val=0)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.01)

    num_epochs = 10
    lr = 0.01
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,None, optimizer)

    X, y = iter(test_iter).next()
    true_labels = d2l.get_fashion_mnish_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnish_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

    d2l.show_fashion_mnist(X[0:9], titles[0:9])

    pass