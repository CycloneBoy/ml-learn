#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : TextAttBiRnn.py
# @Author: sl
# @Date  : 2021/10/30 - 下午9:28


"""
TextAttBiRNN 模型

In the paper [Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems](https://arxiv.org/pdf/1512.08756.pdf), the **feed forward attention** is simplified as follows,

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

from deep.tf.classification.model.other_layers import Attention


class TextAttBiRNN(Model):

    def __init__(self, maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 dropout=0.5,
                 last_activation='softmax'):
        super(TextAttBiRNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.class_num = class_num
        self.dropout = dropout
        self.last_activation = last_activation
        self.embedding = layers.Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
        self.rnn = layers.Bidirectional(
            layers.LSTM(units=embedding_dims, return_sequences=True, dropout=dropout))  # LSTM or GRU
        self.attention = Attention(self.maxlen)
        self.classifier = layers.Dense(class_num, activation=last_activation)

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextAttBiRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError(
                'The maxlen of inputs of TextAttBiRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        emb = self.embedding(inputs)
        y = self.rnn(emb)
        y = self.attention(y)
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
    model = TextAttBiRNN(maxlen=10, max_features=100, embedding_dims=20, class_num=5, )

    model.build_graph(input_shape=(None, 10))
    print(model.summary())
