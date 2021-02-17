#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : house_demo.py
# @Author: sl
# @Date  : 2021/2/15 -  下午3:09


"""
kaggle 房屋销售预测

链接:https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/evaluation

"""

import torch
from time import time
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
import torch.utils.data as Data
from torch import nn
from torch.nn import init
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd

sys.path.append("..")
import deep.torch.d2lzh_pytorch as d2l


def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


def log_rmse(net, features, labels):
    """ 对数均方根误差 """
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay,
          batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    """ KK折交叉验证 """
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls, ['train', 'valid'])
        print('fold %d ,train rmse %f.,valid rmse %f ' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features,test_features,train_labels,test_data,
                   num_epochs,lr,weight_decay,batch_size):
    net = get_net(train_features.shape[1])
    train_ls,_ =train(net,train_features,train_labels,None,None,num_epochs,lr,weight_decay,batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = pd.concat([test_data['Id'],test_data['SalePrice']],axis=1)
    submission.to_csv('./submission.csv',index=False)


if __name__ == '__main__':
    train_data = pd.read_csv('../../../data/txt/train.csv')
    test_data = pd.read_csv('../../../data/txt/test.csv')

    print(train_data.shape)
    print(test_data.shape)
    print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    print(all_features)

    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    print("numeric_features: {}".format(numeric_features))
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))

    print(all_features)
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    print(all_features)
    # print(all_features[0:4,:])

    print(all_features.shape)
    # dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
    all_features = pd.get_dummies(all_features, dummy_na=True)
    print(all_features.shape)
    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
    train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

    loss = torch.nn.MSELoss()

    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    # train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    # print('%d-flod validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))

    # 训练并保存结果
    train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size)
    pass
