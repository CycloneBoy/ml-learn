#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_cuda.py
# @Author: sl
# @Date  : 2020/9/19 - 上午10:22

import torch
print(torch.__version__)

print(torch.version.cuda)
print(torch.backends.cudnn.version())

