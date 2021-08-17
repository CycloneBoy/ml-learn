#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : TextCNN.py
# @Author: sl
# @Date  : 2021/5/19 -  下午4:31

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig

from util.nlp_pretrain import NlpPretrain


class Config(object):

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]
        self.log_path = dataset + '/log/' + self.model_name
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')

        self.require_improvement = 1000
        self.num_classes = len(self.class_list)  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 3
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 5e-5
        self.bert_path = NlpPretrain.BERT_BASE_CHINESE.path
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.model_config = BertConfig.from_pretrained(self.bert_path, output_hidden_states=True)
        self.hidden_size = 768


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, config=config.model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        bert_output = self.bert(context, attention_mask=mask)
        out = self.fc(bert_output.pooler_output)
        return out
