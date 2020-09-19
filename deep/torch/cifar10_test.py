#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : cifar10_test.py
# @Author: sl
# @Date  : 2020/9/19 - 上午9:35


import time


import torch.utils.data
import torchvision
from torch.autograd import Variable

# import torchvision.transforms as transforms
from deep.torch.my_network import LetNet
from util.common_utils import mkdir
from util.constant import CIFAR10_CLASSES
from util.logger_utils import get_log

log = get_log("{}.log".format("cifar10_test"))

# 加载训练集

DATA_CIFAR_DIR = '/home/sl/workspace/data/cifar10'
MODEL_CIFAR_DIR = '{}/model'.format(DATA_CIFAR_DIR)

# #若能使用cuda，则使用cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info("device:{}".format(device))

# 保存模型地址
save_model = "{}/{}".format(MODEL_CIFAR_DIR, "letnet_09190919.pth")

net = torch.load(save_model)
net = net.to(device)
net.eval()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                             std=(0.5, 0.5, 0.5))])


test_dataset = torchvision.datasets.CIFAR10(root=DATA_CIFAR_DIR,
                                            train=False, transform=transform,
                                            download=False)

# 装载数据
batch_size = 100
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)
log.info('len(test_loader) = {}'.format(len(test_loader)))

# 测试
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = net(Variable(images))
        value, predicted = torch.max(preds.data, 1)
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
        _, predicted = torch.max(preds.data, 1)
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