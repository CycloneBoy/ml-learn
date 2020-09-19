#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : my_network.py
# @Author: sl
# @Date  : 2020/9/18 - 下午10:01

import torch
import torch.optim
import torch.utils.data


# LetNet
class LetNet(torch.nn.Module):

    def __init__(self):
        super(LetNet, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 6, kernel_size=5, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(stride=2, kernel_size=2)
        self.conv3 = torch.nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.relu4 = torch.nn.ReLU()
        self.pool5 = torch.nn.MaxPool2d(stride=2, kernel_size=2)
        self.fc6 = torch.nn.Linear(16 * 5 * 5, 120)
        self.relu7 = torch.nn.ReLU()
        self.fc8 = torch.nn.Linear(120, 84)
        self.relu9 = torch.nn.ReLU()
        self.fc10 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu1(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu4(x)
        x = self.pool5(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc6(x)
        x = self.relu7(x)
        x = self.fc8(x)
        x = self.relu9(x)
        x = self.fc10(x)
        return x


# net = LetNet()
# print(net)
