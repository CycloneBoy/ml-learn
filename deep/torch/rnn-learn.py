#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : rnn-learn.py
# @Author: sl
# @Date  : 2020/10/5 - 下午10:18


import torch


def test_add():
    X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
    H, H_hh = torch.randn(3, 4), torch.randn(4, 4)
    res = torch.matmul(X, W_xh) + torch.matmul(H, H_hh)
    print(res)
    ret = torch.matmul(torch.cat((X, H), dim=1), torch.cat((W_xh, H_hh), dim=0))
    print(res)


if __name__ == '__main__':
    test_add()