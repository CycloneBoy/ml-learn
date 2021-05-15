#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : TextRNN.py
# @Author: sl
# @Date  : 2021/5/15 -  下午4:18

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    def __init__(self, dataset, embedding):
        self.model_name = 'TextRNN'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]
        self.vocab_path = dataset + '/data/vocab.pkl'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)['embeddings'].astype('float32')) \
            if embedding != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = 0.5
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)  # 若超过1000batch效果还没提升，则提前结束训练
        self.n_vocab = 0
        self.num_epochs = 20
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度
        self.hidden_size = 128
        self.num_layers = 2


'''
Recurrent Neural Network for Text 
Classification with Multi-Task Learning
'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

    '''变长RNN，效果差不多，甚至还低了点...'''
    def forward1(self, x):
        x, seq_len = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
        _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index

        out = torch.index_select(out, 0, idx_sort)
        seq_len = list(seq_len[idx_sort])
        out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
        #     # [batche_size, seq_len, num_directions * hidden_size]

        out, (hn, _) = self.lstm(out)
        out = torch.cat(hn[2], hn[3], -1)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, atch_first=True)
        out = out.index_select(0, idx_unsort)
        out = self.fc(out[:, -1, :])
        return out
