#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : fashion_mnist.py
# @Author: sl
# @Date  : 2021/2/10 -  上午11:52


"""
  softmax: fashion_mnist

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



if __name__ == '__main__':
    # 加载训练集
    mnist_train = torchvision.datasets.FashionMNIST(root='~/workspace/data/fashionmnist',
                                               train=True, transform=torchvision.transforms.ToTensor(),
                                               download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='~/workspace/data/fashionmnist',
                                              train=False, transform=torchvision.transforms.ToTensor(),
                                              download=True)

    print(type(mnist_train))
    print(len(mnist_train),len(mnist_test))

    feature, label = mnist_train[0]
    print(feature.shape,label)

    X,y = [],[]
    for i in range(10):
        X.append(mnist_train[i][0])
        y.append(mnist_train[i][1])
    # d2l.show_fashion_mnist(X,d2l.get_fashion_mnish_labels(y))

    start = time()
    train_iter,test_iter = d2l.load_data_fashion_mnist()
    for X, y in train_iter:
        continue
    print('%.2f sec' % (time() - start))

    pass


