#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : common_utils.py
# @Author: sl
# @Date  : 2020/9/16 - 下午11:47

import matplotlib.pyplot as plt
import torch
import torchvision

import time
import os



# 创建目录
def mkdir(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

def show_train_image(train_loader):
    images, label = next(iter(train_loader))
    images_example = torchvision.utils.make_grid(images)
    images_example = images_example.numpy().transpose(1, 2, 0)  # 将图像的通道值置换到最后的维度，符合图像的格式
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    images_example = images_example * std + mean
    plt.imshow(images_example)
    plt.show()


# 显示图像１
# show_train_image(train_loader)


def show_train_image2(train_loader):
    for images２, labels２ in train_loader:
        print('image.size() = {}'.format(images２.size()))
        print('labels.size() = {}'.format(labels２.size()))
        break
    # 显示图像２
    # plt.imshow(images[0, 0], cmap='gray')
    # plt.title("label = {}".format(label[0]))


# show_train_image2(train_loader)



if __name__ == '__main__':
    mkdir("/home/sl/workspace/data/test")