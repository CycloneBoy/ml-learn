#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : TextRcnnVariant.py
# @Author: sl
# @Date  : 2021/11/6 - 下午3:20


"""
TextRcnn 模型

RCNN was proposed in the paper Recurrent Convolutional Neural Networks for Text Classification.

Variant of RCNN.

        Base on structure of RCNN, we do some improvement:
        1. Ignore the shift for left/right context.
        2. Use Bidirectional LSTM/GRU to encode context.
        3. Use Multi-CNN to represent the semantic vectors.
        4. Use ReLU instead of Tanh.
        5. Use both AveragePooling and MaxPooling.

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


class TextRcnnVariant(Model):
    """
    Variant of RCNN.

        Base on structure of RCNN, we do some improvement:
        1. Ignore the shift for left/right context.
        2. Use Bidirectional LSTM/GRU to encode context.
        3. Use Multi-CNN to represent the semantic vectors.
        4. Use ReLU instead of Tanh.
        5. Use both AveragePooling and MaxPooling.


    """

    def __init__(self, maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 kernel_sizes=[1, 2, 3, 4, 5],
                 dropout=0.5,
                 kernel_regularizer=None,
                 last_activation='softmax'):
        super(TextRcnnVariant, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.kernel_sizes = kernel_sizes

        self.dropout = dropout
        self.last_activation = last_activation

        self.embedding = layers.Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
        self.bi_rnn = layers.Bidirectional(layers.LSTM(units=embedding_dims, dropout=dropout, return_sequences=True))
        self.concatenate = layers.Concatenate()

        self.conv1s = []
        for kernel_size in self.kernel_sizes:
            self.conv1s.append(layers.Conv1D(filters=128, kernel_size=kernel_size, activation='relu',
                                             kernel_regularizer=kernel_regularizer))

        self.avg_pooling = layers.GlobalAveragePooling1D()
        self.max_pooling = layers.GlobalMaxPooling1D()
        self.classifier = layers.Dense(class_num, activation=last_activation)

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError(
                'The maxlen of inputs of TextRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x_content = self.bi_rnn(embedding)
        x = self.concatenate([embedding, x_content])

        conv1s = []
        for i in range(len(self.kernel_sizes)):
            c = self.conv1s[i](x)  # (batch_size, maxlen-kernel_size+1, filters)
            conv1s.append(c)

        poolings = [self.avg_pooling(conv) for conv in conv1s] + [self.max_pooling(conv) for conv in conv1s]

        x = self.concatenate(poolings)
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
    model = TextRcnnVariant(maxlen=10, max_features=100, embedding_dims=20, class_num=5, )

    model.build_graph(input_shape=(None, 10))
    print(model.summary())
