#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : dropout_simple.py
# @Author: sl
# @Date  : 2021/2/11 -  下午1:34

"""
dropout 简单实现

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
    num_inputs,num_outputs,num_hiddens1,num_hiddens2 = 784,10,256,256
    drop_prob1,drop_prob2 = 0.2,0.5

    net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs,num_hiddens1),
        nn.ReLU(),
        nn.Dropout(drop_prob1),
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        nn.Dropout(drop_prob2),
        nn.Linear(num_hiddens2,num_outputs)
    )

    for param in net.parameters():
        nn.init.normal_(param,mean=0,std=0.01)


    num_epochs,lr,batch_size=5,100.0,256
    optimizer = torch.optim.SGD(net.parameters(), lr)

    loss = torch.nn.CrossEntropyLoss()
    train_iter,test_iter =d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs
                  ,batch_size,None,None,optimizer)

    pass
