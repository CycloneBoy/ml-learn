#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : demo_rnn.py
# @Author: sl
# @Date  : 2021/10/16 - 下午9:45


import tensorflow_datasets as tfds
import tensorflow as tf

from util.constant import IMDB_DATA_DIR



if __name__ == '__main__':
    dataset, info = tfds.load('imdb_reviews', with_info=True, data_dir=IMDB_DATA_DIR, download=False,
                              as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    print(train_dataset.element_spec)