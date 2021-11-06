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
from keras_preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataUtil:

    def __init__(self, maxlen, maxlen_sentence, maxlen_word, max_features,
                 ngram_range=1):
        """

        :param maxlen:
        :param maxlen_sentence:
        :param maxlen_word:
        :param max_features:
        :param ngram_range:  # ngram_range = 2 will add bi-grams features
        """
        self.maxlen = maxlen
        self.maxlen_sentence = maxlen_sentence
        self.maxlen_word = maxlen_word
        self.max_features = max_features

        self.ngram_range = ngram_range

    def load_data_by_name(self, run_model_name):
        if run_model_name == 'han':
            x_train, y_train, x_test, y_test = self.load_data_han(self.maxlen_sentence, self.maxlen_word)
        elif run_model_name == 'textrcnn':
            x_train, y_train, x_test, y_test = self.load_data_rcnn()
        elif run_model_name == 'fasttext':
            x_train, y_train, x_test, y_test = self.load_data_fasttext()
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

    def load_data_fasttext(self):
        print('Loading data...')
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_features)
        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')
        print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
        print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

        if self.ngram_range > 1:
            print('Adding {}-gram features'.format(self.ngram_range))
            # Create set of unique n-gram from the training set.
            ngram_set = set()
            for input_list in x_train:
                for i in range(2, self.ngram_range + 1):
                    set_of_ngram = self.create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            # Dictionary mapping n-gram token to a unique integer.
            # Integer values are greater than max_features in order
            # to avoid collision with existing features.
            start_index = self.max_features + 1
            token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {token_indice[k]: k for k in token_indice}

            # max_features is the highest integer that could be found in the dataset.
            max_features = np.max(list(indice_token.keys())) + 1

            # Augmenting x_train and x_test with n-grams features
            x_train = self.add_ngram(x_train, token_indice, self.ngram_range)
            x_test = self.add_ngram(x_test, token_indice, self.ngram_range)
            print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
            print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

        print('Pad sequences (samples x time)...')
        x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        return x_train, y_train, x_test, y_test

    def create_ngram_set(self, input_list, ngram_value=2):
        """
        Extract a set of n-grams from a list of integers.
        # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
        {(4, 9), (4, 1), (1, 4), (9, 4)}
        # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
        [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
        """
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def add_ngram(self, sequences, token_indice, ngram_range=2):
        """
        Augment the input list of list (sequences) by appending n-grams values.
        Example: adding bi-gram
        # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
        # >>> add_ngram(sequences, token_indice, ngram_range=2)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
        Example: adding tri-gram
        # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
        # >>> add_ngram(sequences, token_indice, ngram_range=3)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
        """
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for ngram_value in range(2, ngram_range + 1):
                for i in range(len(new_list) - ngram_value + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences
