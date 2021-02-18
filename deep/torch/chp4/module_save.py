#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : module_save.py
# @Author: sl
# @Date  : 2021/2/18 -  下午4:34

"""
模型 读取和存储

"""
from collections import OrderedDict

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


class MLP(torch.nn.Module):
    def __init__(self,**kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(3,2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2,1)

    def forward(self,x):
        a = self.act(self.hidden(x))
        return self.output(a)


if __name__ == '__main__':
    x = torch.ones(3)
    torch.save(x,'x.pt')

    x2 = torch.load('x.pt')
    print(x2)

    y = torch.zeros(4)
    torch.save([x,y],'xy.pt')
    xy_list = torch.load('xy.pt')
    print(xy_list)

    torch.save({'x':x,'y':y},'xy_dict.pt')
    xy = torch.load('xy_dict.pt')
    print(xy)

    net = MLP()
    print(net.state_dict())

    optimizer = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    print(optimizer.state_dict())

    X = torch.randn(2,3)
    Y = net(X)

    torch.save(net.state_dict(),'net.pt')

    net1 = MLP()
    net1.load_state_dict(torch.load('net.pt'))
    Y2 = net1(X)
    print(net1)
    print(Y == Y2)

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name())

    x = torch.tensor([1,2,3])
    print(x)

    x = x.cuda(0)
    print(x)
    print(x.device)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x  = torch.tensor([1,2,3],device=device)
    print(x)
    y = x**2
    print(y)
    # z = y +x.cpu()
    net = nn.Linear(3,1)
    res =list(net.parameters())[0].device
    print(res)

    net.cuda()
    res = list(net.parameters())[0].device
    print(res)

    x = torch.rand(2,3).cuda()
    res = net(x)
    print(res)
    pass