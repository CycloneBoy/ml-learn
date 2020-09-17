#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : cifar10_first.py
# @Author: sl
# @Date  : 2020/9/16 - 下午11:01

import matplotlib.pyplot as plt
import torch
import torchvision

from util.common_utils import mkdir

from util.logger_utils import get_log

log = get_log("{}.log".format("cifar10"))

# 加载训练集

DATA_CIFAR_DIR = '/home/sl/workspace/data/cifar10'
MODEL_CIFAR_DIR = '{}/model'.format(DATA_CIFAR_DIR)

mkdir(DATA_CIFAR_DIR)
mkdir(DATA_CIFAR_DIR)

train_dataset = torchvision.datasets.CIFAR10(root=('%s' % DATA_CIFAR_DIR),
                                             train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root=DATA_CIFAR_DIR,
                                            train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)

# 装载数据
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
print('len(train_loader) = {}'.format(len(train_dataset)))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)
print('len(test_loader) = {}'.format(len(test_loader)))


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
show_train_image(train_loader)

for images２, labels２ in train_loader:
    print('image.size() = {}'.format(images２.size()))
    print('labels.size() = {}'.format(labels２.size()))
    break

# 显示图像２
# plt.imshow(images[0, 0], cmap='gray')
# plt.title("label = {}".format(label[0]))

if __name__ == '__main__':
    log.info("test")
