#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : mnist_firtst.py
# @Author: sl
# @Date  : 2020/9/5 - 下午10:45

import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import matplotlib.pyplot as plt
import torch.nn
import torch.optim
import torch


# 加载训练集
train_dataset = torchvision.datasets.MNIST(root='~/workspace/data/mnist',
                                           train=True, transform=torchvision.transforms.ToTensor,
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='~/workspace/data/mnist',
                                           train=False, transform=torchvision.transforms.ToTensor,
                                           download=True)


# 转载数据
batch_size = 100
train_loader =torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size)
print('len(train_loader) = {}'.format(len(train_dataset)))

test_loader =torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size)
print('len(test_loader) = {}'.format(len(test_loader)))

for images, labels in train_loader:
    print('image.size() = {}'.format(images.size()))
    print('labels.size() = {}'.format(labels.size()))
    break


# 显示图像
plt.imshow(images[0,0],cmap='gray')
plt.title("label = {}".format(labels[0]))


# 训练线性分类器

fc = torch.nn.Linear(28 * 28, 10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fc.parameters())

# 训练数据
num_epochs = 5
for epoch in range(num_epochs):
    for idx, (images,labels) in enumerate(train_loader):
        x = images.reshape(-1,28 * 28)

        optimizer.zero_grad()
        preds = fc(x)
        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print('第{}趟第{}批: loss = {:g}'.format(epoch, idx, loss))


# 测试数据
correct = 0
total = 0
for images, labels in test_loader:
    x = images.reshape(-1, 28 * 28)
    preds = fc(x)
    predicted = torch.argmax(preds,1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = correct / total
print('测试集上的准确率:{}'.format(accuracy))



















