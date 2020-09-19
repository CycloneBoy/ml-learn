#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : mnist_nn.py
# @Author: sl
# @Date  : 2020/9/6 - 下午1:41

import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.transforms
import torch.optim
import torchvision.datasets
import torch.nn as nn

import time
import os

from torch.nn.modules import Module, Conv2d, ReLU, Sequential, CrossEntropyLoss


# 创建目录
def mkdir(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)


# 加载训练集
DATA_MNIST_DIR = '~/workspace/data/mnist'
MODEL_MNIST_DIR = '{}/model'.format(DATA_MNIST_DIR)

mkdir(DATA_MNIST_DIR)
mkdir(MODEL_MNIST_DIR)

train_dataset = torchvision.datasets.MNIST(root=('%s' % DATA_MNIST_DIR),
                                           train=True, transform=torchvision.transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root=DATA_MNIST_DIR,
                                          train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 装载数据
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
print('len(train_loader) = {}'.format(len(train_dataset)))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)
print('len(test_loader) = {}'.format(len(test_loader)))


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = torch.nn.ReLU()
        self.pool4 = torch.nn.MaxPool2d(stride=2, kernel_size=2)
        self.fc5 = torch.nn.Linear(128 * 14 * 14, 1024)
        self.relu6 = torch.nn.ReLU()
        self.drop7 = torch.nn.Dropout(p=0.5)
        self.fc8 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu3(x)
        x = self.pool4(x)
        x = x.view(-1, 128 * 14 * 14)
        x = self.fc5(x)
        x = self.relu6(x)
        x = self.drop7(x)
        x = self.fc8(x)
        return x


net = Net()
print(net)


class Net2(torch.nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.conv = Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = Sequential(
            torch.nn.Linear(128 * 14 * 14, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128 * 14 * 14)
        x = self.dense(x)
        return x


net2 = Net2()
print(net2)

criterion = CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

begin_time = time.time()
# 2020/9/6 14:19:33
print('开始训练时间: {}'.format(str(begin_time)))

# 训练
num_epochs = 5
for epoch in range(num_epochs):
    for idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        preds = net(images)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print('epoch {}, batch {},loss = {:g}'.format(epoch, idx, loss.item()))

end_time = time.time()
print('结束训练时间: {}'.format(str(end_time)))
print('训练花费的时间:{} 分钟'.format((end_time - begin_time) / 60))

# 测试
correct = 0
total = 0
for images, labels in test_loader:
    preds = net(images)
    predicted = torch.argmax(preds, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = correct / total
print('测试数据准确率:{:.1%}'.format(accuracy))

# 保存模型
torch.save(net, MODEL_MNIST_DIR)

# 测试数据准确率:98.7%