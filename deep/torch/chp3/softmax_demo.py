#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : softmax_demo.py
# @Author: sl
# @Date  : 2021/2/10 -  下午3:48

"""
softmax 回归 从零开始

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


def test_acc():
    """ 测试 """
    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(X.sum(dim=0, keepdim=True))
    print(X.sum(dim=1, keepdim=True))
    X = torch.rand((2, 5))
    X_prod = d2l.softmax(X)
    print(X_prod, X_prod.sum(dim=1))
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    y = torch.LongTensor([0, 2])
    y_hat.gather(1, y.view(-1, 1))
    print(y_hat, y)
    print(d2l.accuracy(y_hat, y))


def net(X):
    return d2l.softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


if __name__ == '__main__':

    batch_size = 256
    train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs = 28 * 28
    num_outputs = 10
    W = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype=torch.float32)
    b= torch.zeros(num_outputs,dtype=torch.float32)

    W.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)

    # test_acc()

    # net = d2l.softmax_regressive(X,W,b,num_inputs)

    num_epochs = 10
    lr = 0.01
    d2l.train_ch3(net,train_iter,test_iter,d2l.cross_entropy,num_epochs,batch_size,[W,b],lr)

    X,y = iter(test_iter).next()
    true_labels = d2l.get_fashion_mnish_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnish_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true,pred in zip(true_labels,pred_labels)]

    d2l.show_fashion_mnist(X[0:9],titles[0:9])

    pass
