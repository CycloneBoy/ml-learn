#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: sl
# @Date  : 2021/10/23 - 下午3:56


from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.metrics import classification_report
from bert4keras.optimizers import Adam

from data_helper import load_data

# 定义超参数和配置文件
from datamining.multilabel.build_model import build_bert_model
from util.constant import RUN_MODEL

class_nums = 13
maxlen = 60
batch_size = 16

config_path = f'{RUN_MODEL}/bert_config.json'
checkpoint_path = f'{RUN_MODEL}/bert_model.ckpt'
dict_path = f'{RUN_MODEL}/vocab.txt'

tokenizer = Tokenizer(dict_path)


class data_generator(DataGenerator):
    """
    数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)  # [1,3,2,5,9,12,243,0,0,0]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


if __name__ == '__main__':
    # 加载数据集
    train_data = load_data('./data/train.csv')
    test_data = load_data('./data/test.csv')

    # 转换数据集
    train_generator = data_generator(train_data, batch_size)
    test_generator = data_generator(test_data, batch_size)

    model = build_bert_model(config_path, checkpoint_path, class_nums)
    print(model.summary())

    # 冻结参数
    print(f"总层数 {len(model.layers)} ")
    for index, layer in enumerate(model.layers):
        if index <= 104:
            model.layers[index].trainable = False
        print(f" {index} - {layer.name} - {layer.trainable}")
    print("------------")
    print(model.summary())

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(5e-6),
        metrics=['accuracy']
    )

    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=2,
        mode='min'
    )

    best_model_filepath = './checkpoint/best_model.weights'

    checkpoint = keras.callbacks.ModelCheckpoint(
        best_model_filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=test_generator.forfit(),
        validation_steps=len(test_generator),
        shuffle=True,
        callbacks=[earlystop, checkpoint]
    )

    model.load_weights(best_model_filepath)
    test_pred = []
    test_true = []
    for x, y in test_generator:
        p = model.predict(x).argmax(axis=1)
        test_pred.extend(p)

    test_true = test_data[:, 1].tolist()
    print(set(test_true))
    print(set(test_pred))

    target_names = [line.strip() for line in open('label', 'r', encoding='utf8')]
    print(classification_report(test_true, test_pred, target_names=target_names))
