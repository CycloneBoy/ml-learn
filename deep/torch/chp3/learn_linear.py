#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : learn_linear.py
# @Author: sl
# @Date  : 2021/2/9 -  下午9:21

"""
  线性回归: 直接实现

"""

import torch
from time import time
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys

sys.path.append("..")
import deep.torch.d2lzh_pytorch as d2l


# 测试矩阵
def matrix_test():
    a = torch.ones(1000)
    b = torch.ones(1000)
    start = time()
    c = torch.zeros(1000)
    for i in range(1000):
        c[i] = a[i] + b[i]
    print(time() - start)
    start = time()
    d = a + b
    print(time() - start)
    a = torch.ones(3)
    b = 10
    print(a + b)





if __name__ == '__main__':
    # matrix_test()

    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features, labels = d2l.prepare_linear_data(num_examples,num_inputs, true_w, true_b)

    print(features[0], labels[0])

    d2l.set_figsize()
    plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)

    # plt.plot([1, 2, 3, 4])
    # plt.ylabel('some numbers')

    plt.show()

    batch_size = 10
    for X, y in d2l.data_iter(batch_size, features, labels):
        print(X, y)
        break

    w = torch.tensor(np.random.normal(0, 0.000001, (num_inputs, 1)), dtype=torch.float32)
    b = torch.zeros(1, dtype=torch.float32)

    w.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)

    lr = 0.03
    num_epochs = 30
    net = d2l.linreg
    loss = d2l.squared_loss

    for epoch in range(num_epochs):
        for X, y in d2l.data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y).sum()
            l.backward()
            d2l.sgd([w, b], lr, batch_size)

            w.grad.data.zero_()
            b.grad.data.zero_()
        train_l = loss(net(features, w, b), labels)
        print("epoch %d, loss %f" % (epoch + 1, train_l.mean().item()))

    print(true_w, '\n', w)
    print(true_b, '\n', b)
    pass
