#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : BertRCNN.py
# @Author: sl
# @Date  : 2021/5/23 -  下午10:53

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import BertTokenizer, BertModel

from util.nlp_pretrain import NlpPretrain


class Config(object):

    def __init__(self, dataset):
        self.model_name = 'BertRCNN'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]

        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 1000
        self.num_classes = len(self.class_list)  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 3
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 5e-5
        self.bert_path = NlpPretrain.BERT_BASE_CHINESE.path
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

        self.dropout = 0.1
        self.rnn_hidden = 256
        self.num_layers = 1


'''
Recurrent Neural Network for Text 
Classification with Multi-Task Learning
'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)

        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).sequence()
        out = self.fc(out)
        return out

