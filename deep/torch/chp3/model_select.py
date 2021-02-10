#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : model_select.py
# @Author: sl
# @Date  : 2021/2/10 -  下午9:54


"""
模型选择

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

    n_train,n_test,true_w,true_b = 100,100,[1.2,-3.4,5.6],5
    features = torch.randn((n_train + n_test),1)
    ploy_feature = torch.cat((features,torch.pow(features,2),torch.pow(features,3)),1)
    labels = (true_w[0] * ploy_feature[:,0]  + true_w[1] * ploy_feature[:,1]
              + true_w[2] * ploy_feature[:,2] + true_b)
    labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float)

    print(features[:2],ploy_feature[:2],labels[:2])

    # d2l.fit_and_plot(ploy_feature[:n_train,:],ploy_feature[n_train:,:],
    #                  labels[:n_train],labels[n_train:])

    # 线性函数拟合（欠拟合）
    # d2l.fit_and_plot(features[:n_train,:],features[n_train:,:],
    #                  labels[:n_train],labels[n_train:])

    # 训练样本不足（过拟合）
    d2l.fit_and_plot(ploy_feature[:2, :], ploy_feature[n_train:, :],
                     labels[:2], labels[n_train:])

    pass