#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : nlp_pretrain.py
# @Author: sl
# @Date  : 2021/5/19 -  下午11:45
import os
from enum import Enum

from util.constant import NLP_PRETRAIN_DIR


class NlpPretrain(Enum):

    def __init__(self, path, description):
        self.path = path
        self.description = description

    BERT_BASE_UNCASED = (os.path.join(NLP_PRETRAIN_DIR, 'bert-base-uncased'), 'bert')
    BERT_BASE_CHINESE = (os.path.join(NLP_PRETRAIN_DIR, 'bert-base-chinese'), 'bert')
    BERT_CHINESE_WWM = (os.path.join(NLP_PRETRAIN_DIR, 'bert-chinese-wwm'), 'bert-wwm')
    BERT_CHINESE_WWM_EXT = (os.path.join(NLP_PRETRAIN_DIR, 'bert-chinese-wwm-ext'), 'bert-wwm')
    ROBERTA_CHINESE_WWM_EXT_PYTORCH = (os.path.join(NLP_PRETRAIN_DIR, 'chinese_roberta_wwm_ext_pytorch'), 'roberta')
    ROBERTA_CHINESE_WWM_LARGE_EXT_PYTORCH = (
    os.path.join(NLP_PRETRAIN_DIR, 'chinese_roberta_wwm_large_ext_pytorch'), 'roberta')

    def __str__(self):
        return "{}:{}:{}".format(self.name, self.path, self.description)


if __name__ == '__main__':
    print(NlpPretrain.BERT_BASE_CHINESE)