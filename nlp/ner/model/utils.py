#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: sl
# @Date  : 2021/4/18 -  下午2:01

import torch
import torch.nn.functional as F




# ******** LSTM模型 工具函数*************


def cal_loss(logits, targets, tag2id):
    """
    计算损失
    :param logits:  [B, L, out_size]
    :param targets:  [B, L]
    :param tag2id:  [B]
    :return:
    """
    PAD = tag2id.get('<pad>')
    assert  PAD is not None

    mask = (targets != PAD)  # [B, L]
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1,-1,out_size)
    ).contiguous().view(-1,out_size)

    assert logits.size(0) == targets.size(0)

    loss = F.cross_entropy(logits,targets)
    return loss