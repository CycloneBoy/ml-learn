#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : run_multi_class_main.py
# @Author: sl
# @Date  : 2021/7/10 -  下午5:25


import pandas as pd
import xgboost as xgb
import numpy as np

from util.constant import SEEDS_DATA_DIR

params = {
    "objective": "multi:softmax",
    "eta": "0.1",
    "max_depth": 5,
    "num_class": 3
}


def process_data():
    data = pd.read_csv(SEEDS_DATA_DIR, header=None, sep="\s+", converters={7: lambda x: int(x) - 1})
    data.rename(columns={7: 'label'}, inplace=True)
    print(data.head(10))

    mask = np.random.rand(len(data)) < 0.8
    train = data[mask]
    print(mask)
    test = data[~mask]

    xgb_train = xgb.DMatrix(train.iloc[:, :6], label=train.label)
    xgb_test = xgb.DMatrix(test.iloc[:, :6], label=train.label)

    return xgb_train, xgb_test


def main():
    data = pd.read_csv(SEEDS_DATA_DIR, header=None, sep="\s+", converters={7: lambda x: int(x) - 1})
    data.rename(columns={7: 'label'}, inplace=True)
    print(data.head(10))

    mask = np.random.rand(len(data)) < 0.8
    train = data[mask]
    print(mask)
    test = data[~mask]

    xgb_train = xgb.DMatrix(train.iloc[:, :6], label=train.label)
    xgb_test = xgb.DMatrix(test.iloc[:, :6], label=test.label)

    num_round = 50
    watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]

    params['objective'] = "multi:softprob"
    bst = xgb.train(params, xgb_train, num_round, watchlist)
    bst.save_model('./0003.model')

    pred = bst.predict(xgb_test)
    pred_prob = bst.predict(xgb_test)
    print(pred)
    print(pred_prob)

    error_rate = np.sum(pred != test.label) / test.shape[0]
    print("测试集错误率（softmax）：{}".format(error_rate))

    calc_error(pred_prob, test)


def calc_error(pred, test):
    pred_label = np.argmax(pred, axis=1)
    print(pred_label)

    error_rate = np.sum(pred_label != test.label) / test.shape[0]
    print("测试集错误率（softprob）：{}".format(error_rate))


if __name__ == '__main__':
    main()
    # process_data()
