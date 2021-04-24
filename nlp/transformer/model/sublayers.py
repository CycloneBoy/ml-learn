#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : sublayers.py
# @Author: sl
# @Date  : 2021/4/24 -  下午10:22

"""
transformer 中的每个层
"""

import torch
import torch.nn as nn
import torch.nn.init as init

from nlp.transformer.model.modules import Linear
from nlp.transformer.model.modules import ScaledDotProductAttention
from nlp.transformer.model.modules import LayerNormalization


class _MultiHeadAttention(nn.Module):
    def __init__(self,d_k,d_v,d_model,n_heads,dropout):
        super(_MultiHeadAttention,self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = Linear([d_model,d_k * n_heads])
        self.w_k = Linear([d_model,d_k * n_heads])
        self.w_v = Linear([d_model,d_v * n_heads])

        self.attention = ScaledDotProductAttention(d_k,dropout)

    def forward(self,q,k,v,attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        q_s = self.w_q(q).view(b_size,-1,self.n_heads,self.d_k).transpose(1,2)
        k_s = self.w_q(q).view(b_size,-1,self.n_heads,self.d_k).transpose(1,2)
        v_s = self.w_q(q).view(b_size,-1,self.n_heads,self.d_v).transpose(1,2)

        if attn_mask : # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)
        # context: [b_size x n_heads x len_q x d_v], attn: [b_size x n_heads x len_q x len_k]
        context,attn = self.attention(q_s,k_s,v_s,attn_mask=attn_mask)
        # context: [b_size x len_q x n_heads * d_v]
        context = context.transpose(1,2).contiguous().view(b_size,-1,self.n_heads * self.d_v)

        # return the context and attention weights
        return context,attn




