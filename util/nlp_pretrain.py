#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : nlp_pretrain.py
# @Author: sl
# @Date  : 2021/5/19 -  下午11:45
import os
from enum import Enum

from util.constant import NLP_PRETRAIN_DIR


def build_path(name):
    return os.path.join(NLP_PRETRAIN_DIR, name)


class NlpPretrain(Enum):

    def __init__(self, path, description):
        self.path = path
        self.description = description

    BERT_BASE_UNCASED = (build_path('bert-base-uncased'), 'bert')
    BERT_BASE_CHINESE = (build_path('bert-base-chinese'), 'bert')
    BERT_CHINESE_WWM = (build_path('bert-chinese-wwm'), 'bert-wwm')
    BERT_CHINESE_WWM_EXT = (build_path('bert-chinese-wwm-ext'), 'bert-wwm')
    ROBERTA_CHINESE_WWM_EXT_PYTORCH = (build_path('chinese_roberta_wwm_ext_pytorch'), 'roberta')
    ROBERTA_CHINESE_WWM_LARGE_EXT_PYTORCH = (
        build_path('chinese_roberta_wwm_large_ext_pytorch'), 'roberta')

    ELECTRA_CHINESE_SMALL_GENERATOR = (build_path('hfl/chinese-electra-small-generator'), 'electra')
    ELECTRA_CHINESE_SMALL_DISCRIMINATOR = (
        build_path('hfl/chinese-electra-small-discriminator'), 'electra')
    ALBERT_CHINESE_TINY = (
        build_path('voidful/albert_chinese_tiny'), 'albert')

    def __str__(self):
        return "{}:{}:{}".format(self.name, self.path, self.description)


if __name__ == '__main__':
    print(NlpPretrain.BERT_BASE_CHINESE)
