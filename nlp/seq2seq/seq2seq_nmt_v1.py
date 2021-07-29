#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : seq2seq_nmt_v1.py
# @Author: sl
# @Date  : 2021/7/29 -  下午10:07

from torch import nn
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import random
import itertools
import jieba

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


class Voc:

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {
            PAD_token: "<PAD>",
            SOS_token: "/t",
            EOS_token: "/n",
            UNK_token: "<UNK>",
        }
        self.n_words = 4

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def sentence2index(self, sentence):
        index = []
        for word in sentence:
            if word in self.word2index:
                index.append(self.word2index[word])
            else:
                index.append(UNK_token)
        return index


class EncoderGru(nn.Module):
    def __init__(self, hidden_size, embedding):
        super(EncoderGru, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, input_len, hidden):
        embedded = self.embedding(input)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_len)
        output, hidden = self.gru(packed, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        return output, hidden


class DecoderGru(nn.Module):
    def __init__(self, hidden_size, output_size, embedding):
        super(DecoderGru, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = F.relu(embedded)
        output, decoder_hidden = self.gru(output, hidden)
        output = self.softmax(self.fc(output[0]))
        return output, decoder_hidden


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def maskLoss(inp, target, mask):
    crossEntory = torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze())
    loss = crossEntory.masked_select(mask).mean()
    loss = loss.to(device)
    return loss


def sentence2index_eng(voc, sentence):
    return [voc.word2index[word] for word in sentence.split()] + [EOS_token]


def sentence2index_chi(voc, sentence):
    return [voc.word2index[word] for word in jieba.lcut(sentence)] + [EOS_token]


def input_preprocessing(l, voc):
    index_batch = [sentence2index_eng(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(index) for index in index_batch])
    padList = zeroPadding(index_batch)
    padVal = torch.LongTensor(padList)
    return padVal, lengths


def output_preprocessing(l, voc):
    index_batch = [sentence2index_chi(voc, sentence) for sentence in l]
    max_label_len = max([len(index) for index in index_batch])
    padList = zeroPadding(index_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVal = torch.LongTensor(padList)
    return padVal, mask, max_label_len


def data_preprocessing(voc_e, voc_c, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    input, lengths = input_preprocessing(input_batch, voc_e)
    output, mask, max_label_len = output_preprocessing(output_batch, voc_c)
    return input, lengths, output, mask, max_label_len


def train(input_var, input_len, label_var, max_label_len,
          mask, encoder, decoder, encoder_optimizer,
          decoder_optimizer, batch_size):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_var = input_var.to(device)
    label_var = label_var.to(device)
    mask = mask.to(device)

    loss = 0
    encoder_hidden = torch.zeros(1, batch_size, encoder.hidden_size, device=device)  # 初始化编码器的隐层

    _, encoder_hidden = encoder(input_var, input_len, encoder_hidden)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]]).to(device)
    decoder_hidden = encoder_hidden

    for i in range(max_label_len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = torch.LongTensor([[topi[j][0] for j in range(batch_size)]]).to(
            device)  # 每个都取出了最大的,shape=[[tensor1,tensor2,tensor3...]]
        mask_loss = maskLoss(decoder_output, label_var[i], mask[i])
        loss += mask_loss
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item()
