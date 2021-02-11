#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : weight_decay.py
# @Author: sl
# @Date  : 2021/2/11 -  上午11:49


"""
权重衰减

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


def init_params():
    w = torch.randn((num_inputs,1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    return [w,b]


def l2_penalty(w):
    return (w**2).sum()/2


def fit_and_plot(lambd):
    w,b = init_params()
    train_ls,test_ls = [],[]

    for _ in range(num_epochs):
        for X,y in train_iter:
            # 添加了L2范数惩罚项
            l = loss(net(X,w,b),y) + lambd *l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            d2l.sgd([w,b],lr,batch_size)
        train_ls.append(loss(net(train_features,w,b),train_labels).mean().item())
        test_ls.append(loss(net(test_features,w,b),test_labels).mean().item())
    d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','loss',
                    range(1,num_epochs+1),test_ls,['train','loss'])
    print('L2 norm of w:' , w.norm().item())


def fit_and_plot_pytorch(wd):
    net = nn.Linear(num_inputs,1)
    nn.init.normal_(net.weight,mean=0,std=1)
    nn.init.normal_(net.bias,mean=0,std=1)
    optimizer_w = torch.optim.SGD(params=[net.weight],lr=lr,weight_decay=wd)
    optimizer_b = torch.optim.SGD(params=[net.bias],lr=lr)

    train_ls ,test_ls = [],[]
    for _ in range(num_epochs):
        for X,y in train_iter:
            l = loss(net(X),y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            l.backward()

            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'loss'])
    print('L2 norm of w:',net.weight.data.norm().item())


if __name__ == '__main__':

    n_train,n_test,num_inputs = 20,100,200
    true_w,true_b = torch.ones(num_inputs,1) * 0.01 ,0.05

    features = torch.randn((n_train+n_test,num_inputs))
    labels = torch.matmul(features,true_w) + true_b
    labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float)

    train_features,test_features = features[:n_train,:],features[n_train:,:]
    train_labels,test_labels = labels[:n_train],labels[n_train:]


    batch_size ,num_epochs,lr = 1,100,0.003
    net,loss = d2l.linreg,d2l.squared_loss

    dataset = torch.utils.data.TensorDataset(train_features,train_labels)
    train_iter = torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)

    # fit_and_plot(lambd=0)
    # fit_and_plot(lambd=3)
    # fit_and_plot_pytorch(wd=0)
    fit_and_plot_pytorch(wd=3)

    pass