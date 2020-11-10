#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : cifar10_first.py
# @Author: sl
# @Date  : 2020/9/16 - 下午11:01

import time

import torch.utils.data
import torchvision
from torch.autograd import Variable

# import torchvision.transforms as transforms
from deep.torch.my_network import LetNet
from util.common_utils import mkdir
from util.constant import CIFAR10_CLASSES
from util.logger_utils import get_log

log = get_log("{}.log".format("cifar10"))

# 加载训练集

DATA_CIFAR_DIR = '/home/sl/workspace/data/cifar10'
MODEL_CIFAR_DIR = '{}/model'.format(DATA_CIFAR_DIR)

mkdir(DATA_CIFAR_DIR)
mkdir(DATA_CIFAR_DIR)

# 我们将其转化为tensor数据，并归一化为[-1, 1]。
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                             std=(0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root=('%s' % DATA_CIFAR_DIR),
                                             train=True, transform=transform,
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root=DATA_CIFAR_DIR,
                                            train=False, transform=transform,
                                            download=True)

# 装载数据
batch_size = 100
# 将训练集的50000张图片划分成12500份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
log.info('len(train_loader) = {}'.format(len(train_loader)))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)
log.info('len(test_loader) = {}'.format(len(test_loader)))

# #若能使用cuda，则使用cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info("device:{}".format(device))

net = LetNet().to(device)
log.info("net:{}".format(net))

criterion = torch.nn.CrossEntropyLoss()

# 使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

begin_time = time.time()
# 2020/9/6 14:19:33
log.info('开始训练时间: {}'.format(str(begin_time)))

# 训练
num_epochs = 35
for epoch in range(num_epochs):
    running_loss = 0.0
    for idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        images, labels = Variable(images), Variable(labels)

        preds = net(images)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        # 每2000批数据打印一次平均loss值
        running_loss += loss.item()
        if idx % 2000 == 0:
            log.info('epoch {}, batch {},loss = {:g}'.format(epoch, idx, running_loss / 2000))
            running_loss = 0.0

end_time = time.time()
log.info('结束训练时间: {}'.format(str(end_time)))
log.info('训练花费的时间:{} 分钟'.format((end_time - begin_time) / 60))

# 保存模型
save_model = "{}/{}".format(MODEL_CIFAR_DIR, "letnet_09190919.pth")
torch.save(net, save_model)

# 测试
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = net(Variable(images))
        value, predicted = torch.max(preds.response, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
log.info('测试数据准确率:{:.1%}'.format(accuracy))

# 输出10分类每个类别的准确率
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = net(Variable(images))
        _, predicted = torch.max(preds.response, 1)
        c = (predicted == labels).squeeze()
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    class_accuracy = class_correct[i] / class_total[i]
    log.info("测试数据每个类别准确率:{} : {:.1%}".format(CIFAR10_CLASSES[i], class_accuracy))


if __name__ == '__main__':
    pass
