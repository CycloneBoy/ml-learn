#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : multi_net.py
# @Author: sl
# @Date  : 2021/2/10 -  下午9:07


"""
多层感知机 从零开始

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


def net(X):
    """ 两层的 网络"""
    X = X.view((-1,num_inputs))
    H = d2l.relu(torch.matmul(X,W1) + b1)
    return torch.matmul(H,W2) + b2

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 28 * 28
    num_inputs = 784
    num_outputs = 10
    num_hiddens = 256

    W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
    b1 = torch.zeros(num_hiddens, dtype=torch.float)

    W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
    b2 = torch.zeros(num_outputs, dtype=torch.float)

    params = [W1,b1,W2,b2]
    for param in params:
        param.requires_grad_(requires_grad=True)

    loss = torch.nn.CrossEntropyLoss()

    num_epochs = 10
    lr = 100
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

    X, y = iter(test_iter).next()
    true_labels = d2l.get_fashion_mnish_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnish_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

    d2l.show_fashion_mnist(X[0:9], titles[0:9])

    pass
