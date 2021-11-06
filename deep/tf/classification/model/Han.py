#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : Han.py
# @Author: sl
# @Date  : 2021/10/30 - 下午10:03

"""
HAN was proposed in the paper [Hierarchical Attention Networks for Document Classification](http://www.aclweb.org/anthology/N16-1174).

"""

import tensorflow as tf
from keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers

from deep.tf.classification.model.other_layers import Attention

"""
HAN was proposed in the paper Hierarchical Attention Networks for Document Classification.
"""


class HAN(Model):

    def __init__(self, maxlen_sentence,
                 maxlen_word,
                 max_features,
                 embedding_dims,
                 class_num,
                 dropout=0.5,
                 last_activation='sigmod'):
        super(HAN, self).__init__()
        self.maxlen_sentence = maxlen_sentence
        self.maxlen_word = maxlen_word
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.dropout = dropout
        self.last_activation = last_activation
        # Word part
        input_word = Input(shape=(self.maxlen_word,))
        x_word = layers.Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=self.maxlen_word)(
            input_word)
        x_word = layers.Bidirectional(
            layers.LSTM(units=embedding_dims, return_sequences=True, dropout=dropout))(x_word)  # LSTM or GRU
        x_word = Attention(self.maxlen_word)(x_word)
        model_word = Model(input_word, x_word)
        # Sentence part
        self.word_encoder_att = layers.TimeDistributed(model_word)
        self.sentence_encoder = layers.Bidirectional(
            layers.LSTM(units=embedding_dims, return_sequences=True, dropout=dropout))  # LSTM or GRU
        self.sentence_att = Attention(self.maxlen_sentence)
        # Output part
        self.classifier = layers.Dense(class_num, activation=last_activation)

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 3:
            raise ValueError('The rank of inputs of HAN must be 3, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen_sentence:
            raise ValueError('The maxlen_sentence of inputs of HAN must be %d, but now is %d' % (
            self.maxlen_sentence, inputs.get_shape()[1]))
        if inputs.get_shape()[2] != self.maxlen_word:
            raise ValueError('The maxlen_word of inputs of HAN must be %d, but now is %d' % (
            self.maxlen_word, inputs.get_shape()[2]))

        x_sentence = self.word_encoder_att(inputs)
        x_sentence = self.sentence_encoder(x_sentence)
        x_sentence = self.sentence_att(x_sentence)
        output = self.classifier(x_sentence)
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
    model = HAN(maxlen_sentence=10,maxlen_word=25, max_features=100, embedding_dims=20, class_num=5, )

    model.build_graph(input_shape=(None,10,25))
    print(model.summary())