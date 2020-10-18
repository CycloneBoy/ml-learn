#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: sl
# @Date  : 2020/10/16 - 下午11:40

import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter,dev_iter,model,args):
    if args.cuda:
        model.cuda

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    step = 0
    best_acc = 0
    last_step =0
