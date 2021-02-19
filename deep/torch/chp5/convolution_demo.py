#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : convolution_demo.py
# @Author: sl
# @Date  : 2021/2/19 -  上午10:33


"""
卷积神经网络 互相关

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





if __name__ == '__main__':

    X = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
    K = torch.tensor([[0,1],[2,3]])
    res = d2l.corr2d(X,K)
    print(res)

    X = torch.ones(6,8)
    X[:,2:6] = 0
    print(X)

    K = torch.tensor([[1, -1]])
    Y = d2l.corr2d(X, K)
    print(Y)

    pass

