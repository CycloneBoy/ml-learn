#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : module_layer.py
# @Author: sl
# @Date  : 2021/2/18 -  下午4:21


"""
模型 自定义层

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



class CenteredLayer(torch.nn.Module):
    def __init__(self,**kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self,x):
        return x  - x.mean()


class MyDense(torch.nn.Module):
    def __init__(self,**kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4,4)) for i in  range(3)])
        self.params.append(nn.Parameter(torch.randn(4,1)))

    def forward(self,x):
        for i in range(len(self.params)):
            x = torch.mm(x,self.params[i])
        return x


class MyDictDense(torch.nn.Module):
    def __init__(self,**kwargs):
        super(MyDictDense, self).__init__(**kwargs)
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4,4)),
            'linear2': nn.Parameter(torch.randn(4,1))
        })
        self.params.update({'linear3':nn.Parameter(torch.randn(4,2))})

    def forward(self,x,choice='linear1'):
        return torch.mm(x,self.params[choice])


if __name__ == '__main__':
    layer = CenteredLayer()
    res = layer(torch.tensor([1,2,3,4,5],dtype=torch.float))
    print(res)

    net = nn.Sequential(nn.Linear(8,128),CenteredLayer())

    y = net(torch.rand(4,8))
    res = y.mean().item()
    print(res)

    net = MyDense()
    print(net)

    net = MyDictDense()
    print(net)


    x = torch.ones(1,4)
    print(net(x,'linear1'))
    print(net(x,'linear2'))
    print(net(x,'linear3'))

    net = nn.Sequential(
        MyDictDense(),
        MyDense()
    )
    print(net)
    print(net(x))

    pass
