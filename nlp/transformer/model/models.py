#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : models.py
# @Author: sl
# @Date  : 2021/4/25 -  下午9:30

"""
transform 模型
"""

import torch
import torch.nn as nn
import numpy as np
from transformers.modeling_bart import EncoderLayer

from nlp.transformer.model.layers import WeightedEncoderLayer, DecoderLayer, WeightedDecoderLayer
from nlp.transformer.model.modules import PosEncoding
from nlp.transformer.utils import data_utils


def proj_prob_simplex(inputs):
    # project updated weights onto a probability simplex
    # see https://arxiv.org/pdf/1101.6081.pdf
    sorted_inputs, sorted_idx = torch.sort(inputs.view(-1), descending=True)
    dim = len(sorted_inputs)
    for i in reversed(range(dim)):
        t = (sorted_inputs[:i + 1].sum() - 1) / (i + 1)
        if sorted_inputs[i] > t:
            break
    return torch.clamp(inputs - t, min=0.0)


def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(data_utils.PAD).unsqueeze(1)  # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(2)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask


class Encoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, src_vocab_size, dropout=0.1, weighted=False):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=data_utils.PAD)
        self.pos_emb = PosEncoding(max_seq_len * 10, d_model)  # TODO: *10 fix
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = EncoderLayer if not weighted else WeightedEncoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout)
             for _ in range(n_layers)])

    def forward(self, enc_inputs, enc_inputs_len, return_attn=False):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs += self.pos_emb(enc_inputs_len)  # Adding positional encoding TODO: note
        enc_outputs = self.dropout_emb(enc_outputs)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            if return_attn:
                enc_self_attn.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, tgt_vocab_size, dropout=0.1, weighted=False):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=data_utils.PAD)
        self.pos_emb = PosEncoding(max_seq_len * 10, d_model)  # TODO: *10 fix
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = DecoderLayer if not weighted else WeightedDecoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout)
             for _ in range(n_layers)])

    def forward(self, dec_inputs, dec_inputs_len,enc_inputs,enc_outputs, return_attn=False):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs += self.pos_emb(dec_inputs_len)  # Adding positional encoding TODO: note
        dec_outputs = self.dropout_emb(dec_outputs)

        dec_self_attn_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)



        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, dec_self_attn_mask)
            if return_attn:
                enc_self_attn.append(enc_self_attn)

        return enc_outputs, enc_self_attns
