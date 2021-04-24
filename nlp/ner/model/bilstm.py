#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : bilstm.py
# @Author: sl
# @Date  : 2021/4/18 -  下午1:23

"""
双向LSTM模型
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    def __init__(self, vacab_size, emb_size, hidden_size, out_size):
        """
        初始化参数：
        :param vacab_size: 字典的大小
        :param emb_size: 词向量的维数
        :param hidden_size: 隐向量的维数
        :param out_size: 标注的种类
        """
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vacab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)
        self.lin = nn.Linear(2 * hidden_size, out_size)

    def forward(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor)  # [B, L, emb_size]

        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        scores = self.lin(rnn_out)  # [B, L, out_size]
        return scores

    def test(self, sents_tensor, lengths, _):
        """
        第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口
        :param sents_tensor:
        :param lengths:
        :param _:
        :return:
        """
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)
        return batch_tagids

