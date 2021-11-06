#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : FastText.py
# @Author: sl
# @Date  : 2021/11/6 - 下午4:06

"""
FastText was proposed in the paper Bag of Tricks for Efficient Text Classification.
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


class FastText(Model):

    def __init__(self, maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 dropout=0.5,
                 last_activation='softmax'):
        super(FastText, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.dropout = dropout
        self.embedding = layers.Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
        self.avg_pooling = layers.GlobalAveragePooling1D()
        self.classifier = layers.Dense(class_num, activation=last_activation)

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of FastText must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError(
                'The maxlen of inputs of FastText must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        emb = self.embedding(inputs)
        y = self.avg_pooling(emb)
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
    model = FastText(maxlen=10, max_features=100, embedding_dims=20, class_num=5, )

    model.build_graph(input_shape=(None, 10))
    print(model.summary())