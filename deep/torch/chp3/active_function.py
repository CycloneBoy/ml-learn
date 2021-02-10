#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : active_function.py
# @Author: sl
# @Date  : 2021/2/10 -  下午8:52


"""
激活函数

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
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = x.relu()
    d2l.xyplot(x, y, 'relu')

    y.sum().backward()
    d2l.xyplot(x,x.grad,'grad of relu')

    y = x.sigmoid()
    d2l.xyplot(x, y, 'sigmoid')

    x.grad.zero_()
    y.sum().backward()
    d2l.xyplot(x, x.grad, 'grad of sigmoid')

    y = x.tanh()
    d2l.xyplot(x, y, 'tanh')

    x.grad.zero_()
    y.sum().backward()
    d2l.xyplot(x, x.grad, 'grad of tanh')

    pass
