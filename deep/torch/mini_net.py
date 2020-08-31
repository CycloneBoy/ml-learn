#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : mini_net.py
# @Author: sl
# @Date  : 2020/8/31 - 下午11:07

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x),(2,2)))
        x = F.max_pool2d(F.relu(self.conv2(x),2))
        x = x.view(-1,self.num_flat_feature(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_feature(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return  num_features




