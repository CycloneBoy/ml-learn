#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: sl
# @Date  : 2021/10/30 - 下午3:38

"""
训练模型
"""

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ================  params =========================
from deep.tf.classification.model_helper import ModelHelper
from util.constant import IMDB_DATA_DIR

class_num = 2
maxlen = 400
embedding_dims = 200
epochs = 10
batch_size = 128
max_features = 5000

run_model_name = 'TextAttBiRNN'

MODEL_NAME = '{}-epoch-10-emb-200'.format(run_model_name)

use_early_stop = True
tensorboard_log_dir = './logs/{}'.format(MODEL_NAME)
# checkpoint_path = "save_model_dir\\{}\\cp-{epoch:04d}.ckpt".format(MODEL_NAME, '')
checkpoint_path = './save_model_dir/' + MODEL_NAME + '/cp-{epoch:04d}.ckpt'


#  ====================================================================


def load_data():
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print('Pad sequences (samples x time)...')
    x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
    x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()

    model_helper = ModelHelper(class_num=class_num,
                               maxlen=maxlen,
                               max_features=max_features,
                               embedding_dims=embedding_dims,
                               epochs=epochs,
                               batch_size=batch_size,
                               model_name=run_model_name)

    model_helper.get_callback(use_early_stop=use_early_stop, tensorboard_log_dir=tensorboard_log_dir,
                              checkpoint_path=checkpoint_path)

    model_helper.fit(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test)
    print('Test...')
    result = model_helper.model.predict(x_test)
    test_score = model_helper.model.evaluate(x_test, y_test, batch_size=batch_size)

    print("test loss:", test_score[0], "test accuracy", test_score[1])

    model_hepler = ModelHelper(class_num=class_num,
                               maxlen=maxlen,
                               max_features=max_features,
                               embedding_dims=embedding_dims,
                               epochs=epochs,
                               batch_size=batch_size,
                               model_name=run_model_name
                               )
    model_hepler.load_model(checkpoint_path=checkpoint_path)
    # 重新评估模型
    loss, acc = model_hepler.model.evaluate(x_test, y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
