#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_utils.py
# @Author: sl
# @Date  : 2021/11/6 - 下午3:34

"""
数据处理

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataUtil:

    def __init__(self, maxlen, maxlen_sentence, maxlen_word, max_features):
        self.maxlen = maxlen
        self.maxlen_sentence = maxlen_sentence
        self.maxlen_word = maxlen_word
        self.max_features = max_features

    def load_data_by_name(self, run_model_name):
        if run_model_name == 'han':
            x_train, y_train, x_test, y_test = self.load_data_han(self.maxlen_sentence, self.maxlen_word)
        elif run_model_name == 'textrcnn':
            x_train, y_train, x_test, y_test = self.load_data_rcnn()
        else:
            x_train, y_train, x_test, y_test = self.load_data()

        return x_train, y_train, x_test, y_test

    def load_data(self):
        print('Loading data...')
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_features)
        print('Pad sequences (samples x time)...')
        x_train = pad_sequences(x_train, maxlen=self.maxlen, padding='post')
        x_test = pad_sequences(x_test, maxlen=self.maxlen, padding='post')
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)

        return x_train, y_train, x_test, y_test

    def load_data_han(self, maxlen_sentence, maxlen_word):
        print('Loading data for han ...')
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_features)
        print('Pad sequences (samples x time)...')
        pad_maxlen = maxlen_sentence * maxlen_word
        x_train = pad_sequences(x_train, maxlen=pad_maxlen, padding='post')
        x_test = pad_sequences(x_test, maxlen=pad_maxlen, padding='post')
        x_train = x_train.reshape(len(x_train), maxlen_sentence, maxlen_word)
        x_test = x_test.reshape(len(x_test), maxlen_sentence, maxlen_word)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)

        return x_train, y_train, x_test, y_test

    def load_data_rcnn(self):
        x_train, y_train, x_test, y_test = self.load_data()
        print('Prepare input for model rcnn ...')
        x_train_current = x_train
        x_train_left = np.hstack([np.expand_dims(x_train[:, 0], axis=1), x_train[:, 0:-1]])
        x_train_right = np.hstack([x_train[:, 1:], np.expand_dims(x_train[:, -1], axis=1)])
        x_test_current = x_test
        x_test_left = np.hstack([np.expand_dims(x_test[:, 0], axis=1), x_test[:, 0:-1]])
        x_test_right = np.hstack([x_test[:, 1:], np.expand_dims(x_test[:, -1], axis=1)])
        print('x_train_current shape:', x_train_current.shape)
        print('x_train_left shape:', x_train_left.shape)
        print('x_train_right shape:', x_train_right.shape)
        print('x_test_current shape:', x_test_current.shape)
        print('x_test_left shape:', x_test_left.shape)
        print('x_test_right shape:', x_test_right.shape)

        return [x_train_current, x_train_left, x_train_right], y_train, [x_test_current, x_test_left,
                                                                         x_test_right], y_test
