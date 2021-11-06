#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : Rcnn.py
# @Author: sl
# @Date  : 2021/11/6 - 下午2:40


"""
TextRcnn 模型

RCNN was proposed in the paper Recurrent Convolutional Neural Networks for Text Classification.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


class TextRCNN(Model):

    def __init__(self, maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 dropout=0.5,
                 last_activation='softmax'):
        super(TextRCNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.dropout = dropout
        self.last_activation = last_activation

        self.embedding = layers.Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
        self.forward_rnn = layers.SimpleRNN(units=embedding_dims, dropout=dropout, return_sequences=True)
        self.backward_rnn = layers.SimpleRNN(units=embedding_dims, dropout=dropout, return_sequences=True, go_backwards=True)
        self.reverse = layers.Lambda(lambda x: tf.reverse(x, axis=[1]))
        self.concatenate = layers.Concatenate(axis=2)
        self.conv = layers.Conv1D(64, kernel_size=1, activation='tanh')
        self.max_pooling = layers.GlobalMaxPooling1D()

        self.classifier = layers.Dense(class_num, activation=last_activation)

    def call(self, inputs, training=None, mask=None):
        if len(inputs) != 3:
            raise ValueError('The length of inputs of RCNN must be 3, but now is %d' % len(inputs))
        input_current = inputs[0]
        input_left = inputs[1]
        input_right = inputs[2]
        if len(input_current.get_shape()) != 2 or len(input_left.get_shape()) != 2 or len(input_right.get_shape()) != 2:
            raise ValueError('The rank of inputs of RCNN must be (2, 2, 2), but now is (%d, %d, %d)' % (
                len(input_current.get_shape()), len(input_left.get_shape()), len(input_right.get_shape())))
        if input_current.get_shape()[1] != self.maxlen or input_left.get_shape()[1] != self.maxlen or \
                input_right.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of RCNN must be (%d, %d, %d), but now is (%d, %d, %d)' % (
                self.maxlen, self.maxlen, self.maxlen, input_current.get_shape()[1], input_left.get_shape()[1],
                input_right.get_shape()[1]))
        embedding_current = self.embedding(input_current)
        embedding_left = self.embedding(input_left)
        embedding_right = self.embedding(input_right)
        x_left = self.forward_rnn(embedding_left)
        x_right = self.forward_rnn(embedding_right)
        x_right = self.reverse(x_right)
        x = self.concatenate([x_left, embedding_current, x_right])
        x = self.conv(x)
        x = self.max_pooling(x)
        output = self.classifier(x)
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
    model = TextRCNN(maxlen=10, max_features=100, embedding_dims=20, class_num=5, )

    model.build_graph(input_shape=(None, 3, 10))
    print(model.summary())
