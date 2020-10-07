#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : lstm-learn.py
# @Author: sl
# @Date  : 2020/10/7 - 下午10:43

import sys

sys.path.append("..")

import zipfile

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from util.logger_utils import get_log

log = get_log("{}.log".format("lstm-learn"))


if __name__ == '__main__':
    # res = read_jay()
    # log.info("{}".format(res))

    # res = index_jay()
    # log.info("{}".format(res))
    pass
