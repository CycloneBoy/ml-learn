#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : model_helper.py
# @Author: sl
# @Date  : 2021/10/30 - 下午3:07
import shutil

import keras.callbacks
import numpy as np
import tensorflow as tf
import random as rn

from deep.tf.classification.model.FastText import FastText
from deep.tf.classification.model.Han import HAN
from deep.tf.classification.model.Rcnn import TextRCNN
from deep.tf.classification.model.TextAttBiRnn import TextAttBiRNN
from deep.tf.classification.model.TextCnn import TextCNN
from deep.tf.classification.model.TextRcnnVariant import TextRcnnVariant
from deep.tf.classification.model.TextRnn import TextRNN

np.random.seed(0)
rn.seed(0)
tf.random.set_seed(0)

import os


def checkout_dir(dir_path, do_delete=False):
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)


class ModelHelper:

    def __init__(self, class_num, maxlen, max_sentence, maxlen_word, max_features, embedding_dims, epochs, batch_size,
                 model_name):
        self.class_num = class_num
        self.maxlen = maxlen
        self.max_sentence = max_sentence
        self.maxlen_word = maxlen_word
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.callback_list = []
        self.model_name = str(model_name).lower()
        print('Bulid Model: {} ...'.format(model_name))
        self.create_model(model_name)

    def create_model(self, model_name):
        if model_name == "textrnn":
            model = TextRNN(maxlen=self.maxlen,
                            max_features=self.max_features,
                            embedding_dims=self.embedding_dims,
                            class_num=self.class_num,
                            last_activation='softmax')
        elif model_name == "textattbirnn":
            model = TextAttBiRNN(maxlen=self.maxlen,
                                 max_features=self.max_features,
                                 embedding_dims=self.embedding_dims,
                                 class_num=self.class_num,
                                 last_activation='softmax')

        elif model_name == "han":
            model = HAN(maxlen_sentence=self.max_sentence,
                        maxlen_word=self.maxlen_word,
                        max_features=self.max_features,
                        embedding_dims=self.embedding_dims,
                        class_num=self.class_num,
                        last_activation='softmax')
        elif model_name == "textrcnn":
            model = TextRCNN(maxlen=self.maxlen,
                             max_features=self.max_features,
                             embedding_dims=self.embedding_dims,
                             class_num=self.class_num,
                             last_activation='softmax')
        elif model_name == "textrcnn_variant":
            model = TextRcnnVariant(maxlen=self.maxlen,
                             max_features=self.max_features,
                             embedding_dims=self.embedding_dims,
                             class_num=self.class_num,
                             kernel_regularizer=None,
                             last_activation='softmax')
        elif model_name == "fasttext":
            model = FastText(maxlen=self.maxlen,
                             max_features=self.max_features,
                             embedding_dims=self.embedding_dims,
                             class_num=self.class_num,
                             last_activation='softmax')
        else:
            model = TextCNN(maxlen=self.maxlen,
                            max_features=self.max_features,
                            embedding_dims=self.embedding_dims,
                            class_num=self.class_num,
                            kernel_sizes=[2, 3, 5],
                            kernel_regularizer=None,
                            last_activation='softmax')

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        model.build_graph(input_shape=(None, self.maxlen))
        model.summary()
        self.model = model

    def get_callback(self, use_early_stop=True,
                     tensorboard_log_dir="logs",
                     checkpoint_path="save_model",
                     ):
        callback_list = []
        if use_early_stop:
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, mode='max')
            callback_list.append(early_stopping)

        if checkpoint_path is not None:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkout_dir(checkpoint_dir, do_delete=True)
            cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                          monitor='val_accuracy',
                                                          mode='max',
                                                          save_best_only=True,
                                                          save_weights_only=True,
                                                          verbose=1,
                                                          period=2)
            callback_list.append(cp_callback)

        if tensorboard_log_dir is not None:
            checkout_dir(tensorboard_log_dir, do_delete=True)
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
            callback_list.append(tensorboard_callback)

        self.callback_list = callback_list

    def fit(self, x_train, y_train, x_val, y_val):
        print("Train...")
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=2,
                       callbacks=self.callback_list,
                       validation_data=(x_val, y_val))

    def load_model(self, checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print('restore model name is : ', latest)
        # 创建一个新的模型实例
        # model = self.create_model()
        # 加载以前保存的权重
        self.model.load_weights(latest)
