#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : module_init.py
# @Author: sl
# @Date  : 2021/2/18 -  下午2:22


"""
模型 参数初始化

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

class MyModule(torch.nn.Module):
    def __init__(self,**kwargs):
        super(MyModule, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20,20))
        self.weight2 = torch.rand(20,20)

    def forward(self,x):
        pass

def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10,10)
        tensor *= (tensor.abs() >= 5).float()



if __name__ == '__main__':

    net = nn.Sequential(nn.Linear(4,3),nn.ReLU(),nn.Linear(3,1))

    print(net)
    X = torch.rand(2,4)
    Y = net(X).sum()
    print(Y)

    print(type(net.named_parameters()))
    for name ,param in net.named_parameters():
        print(name,param.size())

    for name ,param in net[0].named_parameters():
        print(name,param.size(),type(param))

    n = MyModule()
    for name ,param in n.named_parameters():
        print(name)

    weight_0 = list(net[0].parameters())[0]
    print(weight_0.data)
    print(weight_0.grad)
    Y.backward()
    print(weight_0.grad)

    for name ,param in net.named_parameters():
        if 'weight' in name:
            init.normal_(param,mean=0,std=0.01)
            print(name,param.data)

    for name,param in net.named_parameters():
        if 'bias' in name:
            init.constant_(param,val=0)
            print(name,param.data)


    for name ,param in net.named_parameters():
        if 'weight' in name:
            init_weight_(param)
            print(name,param.data)


    for name ,param in net.named_parameters():
        if 'bias' in name:
            param.data +=1
            print(name,param.data)


    linear = nn.Linear(1,1,bias=False)
    net = nn.Sequential(linear,linear)
    print(net)
    for name,param in net.named_parameters():
        init.constant_(param,val=3)
        print(name,param.data)

    print(id(net[0]) == id(net[1]))
    print(id(net[0].weight) == id(net[1].weight))

    x = torch.ones(1,1)
    y = net(x).sum()
    print(y)
    y.backward()
    print(net[0].weight.grad)
    pass