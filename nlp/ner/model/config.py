#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : config.py
# @Author: sl
# @Date  : 2021/4/18 -  下午1:57

"""
模型的配置文件
"""


class TrainingConfig(object):
    batch_size = 64
    # 学习率
    lr = 0.001
    epoches = 30
    print_step = 5


class LSTMConfig(object):
    emb_size = 128  # 词向量的维数
    hidden_size = 128  # lstm隐向量的维数
