#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : text_sentiment_classification.py
# @Author: sl
# @Date  : 2020/10/11 - 下午4:50

import collections
import os
import random
import tarfile
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data

import sys
sys.path.append("..")
import deep.torch.d2lzh_pytorch as d2l


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ROOT = "/S1/CSCL/tangss/Datasets"
