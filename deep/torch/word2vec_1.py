#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : word2vec_1.py
# @Author: sl
# @Date  : 2020/10/11 - 上午11:40


import collections
import math
import random
import sys
import time

import torch
import torch.utils.data as Data
from torch import nn

sys.path.append("..")

print(torch.__version__)


def read_ptb():
    with open('../../data/test/ptb/ptb.train.txt', 'r') as f:
        lines = f.readlines()
        raw_dataset = [st.split() for st in lines]

    print('# sentences: %d' % len(raw_dataset))
    return raw_dataset


def print_ptd():
    raw_dataset = read_ptb()
    for st in raw_dataset[:3]:
        print(st)
        print('# tokens:', len(st), st[:5])


def index_sent():
    raw_dataset = read_ptb()
    counter = collections.Counter([tk for st in raw_dataset for tk in st])
    counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
    idx_to_token = [tk for tk, _ in counter.items()]
    token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
    dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx] for st in raw_dataset]
    num_tokens = sum([len(st) for st in dataset])
    print('# tokens:%d' % num_tokens)  # 输出 '# tokens: 887100'
    return counter, idx_to_token, token_to_idx, dataset, num_tokens


counter, idx_to_token, token_to_idx, dataset, num_tokens = index_sent()


def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens)


subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
print('# tokens:%d ' % sum([len(st) for st in subsampled_dataset]))


def compare_counts(token):
    return '# %s: before=%d, after=%d ' % (token, sum(
        [st.count(token_to_idx[token]) for st in dataset]), sum(
        [st.count(token_to_idx[token]) for st in subsampled_dataset]))


def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts


def test_tiny_dataset():
    tiny_dataset = [list(range(7)), list(range(7, 10))]
    print(tiny_dataset)
    for center, contex in zip(*get_centers_and_contexts(tiny_dataset, 2)):
        print('center', center, 'has coontexts', contex)


def test_compare():
    res = compare_counts('the')
    print(res)
    res = compare_counts('join')
    print(res)


all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)


# 负采样
# 我们使用负采样来进行近似训练。对于一对中心词和背景词，我们随机采样KK个噪声词（实验中设K=5K=5）。根据word2vec论文的建议，噪声词采样概率P(w)P(w)设为ww词频与总词频之比的0.75次方 [2]。
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                # 为了高效计算，可以将k设得稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


sampling_weights = [counter[w] ** 0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])

    def __len__(self):
        return len(self.centers)


def batchify(data):
    """用作DataLoader的参数collate_fn: 输入是个长为batchsize的list,
        list中的每个元素都是Dataset类调用__getitem__得到的结果
    """
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives),
            torch.tensor(masks), torch.tensor(labels))


batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 8

dataset = MyDataset(all_centers, all_contexts, all_negatives)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True, collate_fn=batchify, num_workers=num_workers)

for batch in data_iter:
    for name, data in zip(['centers', 'contexts_negatives', 'masks', 'labels'], batch):
        print(name, 'shape:', data.shape)
    break

embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(embed.weight)

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
res = embed(x)
print(res)


def test_bmm():
    X = torch.ones((2, 1, 4))
    Y = torch.ones((2, 4, 6))
    res = torch.bmm(X, Y).shape
    print(res)


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        """
                 input – Tensor shape: (batch_size, len)
                target – Tensor of the same shape as input
                """
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        return res.mean(dim=1)


loss = SigmoidBinaryCrossEntropyLoss()


def test_loss():
    pred = torch.tensor([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
    # 标签变量label中的1和0分别代表背景词和噪声词
    label = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])  # 掩码变量
    res = loss(pred, label, mask) * mask.shape[1] / mask.float().sum(dim=1)
    print(res)


def sigmd(x):
    return - math.log(1 / (1 + math.exp(-x)))


print('%.4f' % ((sigmd(1.5) + sigmd(-0.3) + sigmd(1) + sigmd(-2)) / 4))  # 注意1-sigmoid(x) = sigmoid(-x)
print('%.4f' % ((sigmd(1.1) + sigmd(-0.6) + sigmd(-2.2)) / 3))

embed_size = 100
net = nn.Sequential(
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size)
)


def train(net, lr, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('train on', device)

    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start, l_sum, n, l_sum_gpu = time.time(), 0.0, 0, 0.0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for d in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            # 使用掩码变量mask来避免填充项对损失函数计算的影响
            l = (loss(pred.view(label.shape), label, mask) *
                 mask.shape[1] / mask.float().sum(dim=1)).mean()  # 一个batch的平均loss

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            l_sum_gpu += l.item()
            n += 1
        print('epoch %d, loss %.2f,loss_gpu %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, l_sum_gpu / n, time.time() - start))


def get_similar_tokens(query_token, k, embed):
    W = embed.weight.url
    x = W[token_to_idx[query_token]]
    # 添加的1e-9是为了数值稳定性
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
    _, topk = torch.topk(cos, k=k + 1)
    topk = topk.cpu().numpy()
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i], (idx_to_token[i])))


if __name__ == '__main__':
    # read_ptb()
    # print_ptd()
    # index_sent()
    # test_compare()
    # test_tiny_dataset()
    # test_bmm()
    # test_loss()
    train(net, 0.01, 10)
    get_similar_tokens('chip', 3, net[0])
    pass
