#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : common_utils.py
# @Author: sl
# @Date  : 2020/9/16 - 下午11:47

import time
import os



# 创建目录
def mkdir(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)


if __name__ == '__main__':
    mkdir("/home/sl/workspace/data/test")