#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_torch.py
# @Author: sl
# @Date  : 2020/8/22 - 上午8:52

from __future__ import print_function
import torch
import torchvision


def run():
    x = torch.rand(5, 3)
    print(x)
    print(torch.cuda.is_available())


if __name__ == '__main__':
    run()
    print(torchvision.__version__)
    print(torch.__version__)
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))
