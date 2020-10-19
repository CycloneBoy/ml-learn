#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : common_utils.py
# @Author: sl
# @Date  : 2020/9/16 - 下午11:47

import matplotlib.pyplot as plt
import torch
import torchvision

import torchtext.vocab as vocab

from util.constant import DATA_CACHE_DIR, GLOVE_DATA_DIR, MODEL_NLP_DIR

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

# 获取glove 词嵌入
def get_glove(name='6B',dim=100):
    name = "glove.{}.{}d".format(name,dim)
    glove = vocab.pretrained_aliases[name](cache=GLOVE_DATA_DIR)
    return glove

def knn(W, x, k):
    # 添加的1e-9是为了数值稳定性
    cos = torch.matmul(W, x.view((-1,))) / ((torch.sum(W * W, dim=1) + 1e-9).sqrt() * torch.sum(x * x).sqrt())
    _, topk = torch.topk(cos, k=k)
    topk = topk.cpu().numpy()
    return topk, [cos[i].item() for i in topk]



def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.vectors, embed.vectors[embed.stoi[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # 除去输入词
        print('cosine sim=%.3f: %s' % (c, (embed.itos[i])))
    # return (c)


# 求类比词 求类比词问题可以定义为：对于类比关系中的4个词 a:b::c:da:b::c:d，给定前3个词aa、bb和cc，求dd。设词ww的词向量为vec(w)vec(w)。求类比词的思路是，搜索与vec(c)+vec(
# b)−vec(a)vec(c)+vec(b)−vec(a)的结果向量最相似的词向量。
def get_analogy(token_a, token_b, token_c, embed):
    vecs = [embed.vectors[embed.stoi[t]] for t in [token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.vectors, x, 1)
    return embed.itos[topk[0]]

def save_model(net,filename="model1.pth",datadir=MODEL_NLP_DIR):
    save_model = "{}/{}".format(datadir, filename)
    print("保存模型:",save_model)
    torch.save(net, save_model)

if __name__ == '__main__':
    # mkdir("/home/sl/workspace/data/test")

    glove = get_glove()
    print("一共包含%d个词" % len(glove.stoi))

    print(glove.stoi['beautiful'])
    print(glove.itos[3366])