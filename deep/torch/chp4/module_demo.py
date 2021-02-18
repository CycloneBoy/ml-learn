#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : module_demo.py
# @Author: sl
# @Date  : 2021/2/18 -  下午1:36


"""
模型构造 从零开始

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


class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


class Module_ModuleList(torch.nn.Module):
    def __init__(self):
        super(Module_ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10)])


class Module_List(torch.nn.Module):
    def __init__(self):
        super(Module_List, self).__init__()
        self.linears = [nn.Linear(10, 10)]


class FancyMLP(torch.nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        x = self.linear(x)
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.0:
            x *= 10
        return x.sum()


class NestMLP(torch.nn.Module):
    def __init__(self,**kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40,30),nn.ReLU())

    def forward(self,x):
        return self.net(x)

    

if __name__ == '__main__':

    X = torch.rand(2, 784)
    net = MLP()
    print(net)
    print(net(X))

    net = MySequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    print(net)
    print(net(X))

    net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
    net.append(nn.Linear(256, 10))
    print(net[-1])
    print(net)
    # net(X)

    net1 = Module_ModuleList()
    net2 = Module_List()

    print("net1")
    for p in net1.parameters():
        print(p.size())

    print("net2")
    for p in net2.parameters():
        print(p)

    net = nn.ModuleDict({
        'linear': nn.Linear(784, 256),
        'act': nn.ReLU(),
    })
    net['output'] = nn.Linear(256, 10)
    print(net['linear'])
    print(net.output)
    print(net)

    x = torch.rand(2,20)
    net = FancyMLP()
    print(net)
    res = net(x)
    print(res)

    net = nn.Sequential(NestMLP(),nn.Linear(30,20),FancyMLP())

    X = torch.rand(2,40)
    print(net)
    res = net(X)
    print(res)
    pass
