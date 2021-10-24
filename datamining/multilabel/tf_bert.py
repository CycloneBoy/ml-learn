#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : tf_bert.py
# @Author: sl
# @Date  : 2021/10/23 - 上午11:40

"""
tf bert learn
"""

import tensorflow as tf


from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.snippets import to_array
from bert4keras.tokenizers import Tokenizer
import numpy as np

from util.constant import RUN_MODEL


def print_bert_variables():
    for v in tf.train.list_variables(f'{RUN_MODEL}/bert_model.ckpt'):
        print(v)


def bert_demo():
    config_path = f'{RUN_MODEL}/bert_config.json'
    checkpoint_path = f'{RUN_MODEL}/bert_model.ckpt'
    dict_path = f'{RUN_MODEL}/vocab.txt'
    tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
    model = build_transformer_model(config_path, checkpoint_path, with_mlm=True)  # 建立模型，加载权重
    # 编码测试
    token_ids, segment_ids = tokenizer.encode(u'语言模型')
    print('\n ===== predicting =====\n')
    print(model.predict([np.array([token_ids]), np.array([segment_ids])]))

    # model.save('test.model')
    # del model
    # model = keras.models.load_model('test.model')
    # print(model.predict([token_ids, segment_ids]))

    token_ids, segment_ids = tokenizer.encode(u'科学技术是第一生产力')

    # mask掉“技术”
    token_ids[3] = token_ids[4] = tokenizer._token_mask_id
    token_ids, segment_ids = to_array([token_ids], [segment_ids])

    # 用mlm模型预测被mask掉的部分
    probas = model.predict([token_ids, segment_ids])[0]
    print(tokenizer.decode(probas[3:5].argmax(axis=1)))  # 结果正是“技术”


def bert_class():
    pass


if __name__ == '__main__':
    pass

    bert_demo()

    # print_bert_variables()
