#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : TextRnn.py
# @Author: sl
# @Date  : 2021/10/30 - 下午4:26

"""
TextRnn 模型
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


class TextRNN(Model):

    def __init__(self, maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 dropout=0.5,
                 last_activation='softmax'):
        super(TextRNN, self).__init__()
        self.maxlen = maxlen
        self.class_num = class_num
        self.dropout = dropout
        self.embedding = layers.Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
        self.rnn = layers.Bidirectional(layers.LSTM(units=embedding_dims, dropout=dropout))
        self.classifier = layers.Dense(class_num, activation=last_activation)

    def call(self, inputs, training=None, mask=None):
        emb = self.embedding(inputs)
        y = self.rnn(emb)
        output = self.classifier(y)
        return output

    def build_graph(self, input_shape):
        '''
        自定义函数，在调用model.summary()之前调用
        '''
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)

if __name__ == '__main__':
    model = TextRNN(maxlen=10, max_features=100, embedding_dims=20, class_num=5, )

    model.build_graph(input_shape=(None,10))
    print(model.summary())
