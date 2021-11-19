#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : TextBert.py
# @Author: sl
# @Date  : 2021/11/6 - 下午5:17

"""
Bert Model

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from transformers import TFBertModel


class TextBert(Model):

    def __init__(self, bert: TFBertModel, class_num: int,
                 last_activation='softmax'):
        """

        :param bert:
        :param num_classes:
        :param last_activation:
        """
        super().__init__()
        self.class_num = class_num
        self.last_activation = last_activation
        self.bert = bert
        self.classifier = layers.Dense(class_num, activation=last_activation)

    @tf.function
    def call(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            )
        cls_output = outputs[1]
        cls_output = self.classifier(cls_output)
        return cls_output
