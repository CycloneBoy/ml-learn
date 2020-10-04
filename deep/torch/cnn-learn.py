#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : cnn-learn.py
# @Author: sl
# @Date  : 2020/10/3 - 下午9:36

# https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter05_CNN/5.4_pooling

import sys

sys.path.append("..")

import torch
from torch import nn

import deep.torch.d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log = d2l.log


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
res = corr2d(X, K)
print(res)


class Conv2D(nn.Module):

    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1, -1]])

Y = corr2d(X, K)
print(Y)

# 5.1.4 通过数据学习核数组

conv2d = Conv2D(kernel_size=(1, 2))

step = 50
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()

    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print('step %d ,loss %.3f' % (i + 1, l.item()))

print("weight: ", conv2d.weight.data)
print("bias: ", conv2d.bias.data)


def comp_conv2d(conv2d, X):
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])


conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

X = torch.rand(8, 8)
x_shape = comp_conv2d(conv2d, X).shape
print(x_shape)

# 使用高为5、宽为3的卷积核。在高和宽两侧的填充数分别为2和1
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
x_shape = comp_conv2d(conv2d, X).shape
print(x_shape)

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
x_shape = comp_conv2d(conv2d, X).shape
print(x_shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
x_shape = comp_conv2d(conv2d, X).shape
print(x_shape)


# 5.3.1 多输入通道

def corr2d_multi_in(X, K):
    res = corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += corr2d(X[i, :, :], K[i, :, :])
    return res


X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

res = corr2d_multi_in(X, K)
print(res)


# 5.3.2 多输出通道

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K])


K = torch.stack([K, K + 1, K + 2])
print(K.shape)

res = corr2d_multi_in_out(X, K)
print(res)


# 5.3.3 1×11×1卷积层

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X)
    return Y.view(c_o, h, w)


X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

res = (Y1 - Y2).norm().item() < 1e-6
print(res)


def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
res = pool2d(X, (2, 2))
print(res)

X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
print(X)

pool2d = nn.MaxPool2d(3)
res = pool2d(X)
print(res)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
res = pool2d(X)
print(res)

pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
res = pool2d(X)
print(res)

X = torch.cat((X, X + 1), dim=1)
print(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
res = pool2d(X)
print(res)


def test_letnet():
    # LetNet
    log.info("测试　LetNet")
    net = d2l.LetNet()
    log.info(net)
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


def test_alexnet():
    # AlexNet
    log.info("测试　AlexNet")
    net = d2l.AlexNet()
    log.info(net)
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 这时候我们可以开始训练AlexNet了。相对于LeNet，由于图片尺寸变大了而且模型变大了，所以需要更大的显存，也需要更长的训练时间了。
    batch_size = 32
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


def test_vgg11():
    log.info("测试　vgg11")
    net = d2l.vgg11_small()
    log.info(net)
    batch_size = 64
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

def test_nin_net():
    log.info("测试　nin_net")
    net = d2l.nin_net()
    log.info(net)
    batch_size = 64
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.002, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


def test_google_net():
    log.info("测试　google_net")
    net = d2l.GoogLeNet()
    log.info(net)
    batch_size = 16
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


def test_letnet_batch_normal():
    log.info("测试　letnet_batch_normal")
    net = d2l.LetNetWithBatchNormal()
    log.info(net)
    batch_size = 256
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    # print(net[1].gamma.view((-1,)), net[1].beta.view((-1,)))

def test_letnet_torch_batch_normal():
    log.info("测试　letnet_batch_normal")
    net = d2l.LetNetWithTorchBatchNormal()
    log.info(net)
    batch_size = 256
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    # print(net[1].gamma.view((-1,)), net[1].beta.view((-1,)))


def test_residual():
    blk = d2l.Residual(3, 3)
    X = torch.rand((4,3,6,6))
    res = blk(X).shape
    print(res)

    blk = d2l.Residual(3, 6,use_1x1conv=True,stride=2)
    X = torch.rand((4,3,6,6))
    res = blk(X).shape
    print(res)


def test_resnet():
    log.info("测试　resnet")
    net = d2l.ResNet18()
    log.info(net)
    batch_size = 256
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    # print(net[1].gamma.view((-1,)), net[1].beta.view((-1,)))


def test_dense_net():
    log.info("测试　dense_net")
    net = d2l.DenseNet()
    log.info(net)
    batch_size = 256
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    # print(net[1].gamma.view((-1,)), net[1].beta.view((-1,)))


if __name__ == '__main__':
    # test_letnet()
    # test_alexnet()
    # test_vgg11()
    # test_nin_net()
    # test_google_net()
    # test_letnet_batch_normal()
    # test_letnet_torch_batch_normal()
    # test_residual()
    # test_resnet()
    test_dense_net()