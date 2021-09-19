#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : lr.py
# @Author: sl
# @Date  : 2021/9/19 - 下午9:06

import torch

from deep.ctr.model.layer import FeaturesLinear


class LogisticRegressionModel(torch.nn.Module):
    """
    A pytorch implementation of Logistic Regression.
    """

    def __init__(self, field_dims):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return torch.sigmoid(self.linear(x).squeeze(1))
