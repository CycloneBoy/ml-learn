#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : bbox.py
# @Author: sl
# @Date  : 2020/10/3 - 下午2:56

import torch
import torchvision


image = torch.zeros((1,3,800,800)).float()

bbox = torch.FloatTensor([[200,30,400,500],[300,400,500,600]])

#[y1,x1,y2,x2] format
label = torch.LongTensor([6,8])
sub_sample = 16


# 1. 生成一个dummy image并且设置volatile为False：
dummy_img = torch.zeros((1, 3, 800, 800)).float()
print(dummy_img.size())

# 2. 列出VGG16的所有层：
model = torchvision.models.vgg16(pretrained=True)
fe = list(model.featurese)
print(fe)

