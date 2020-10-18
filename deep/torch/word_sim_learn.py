#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : word_sim_learn.py
# @Author: sl
# @Date  : 2020/10/11 - 下午4:16

import torch
import torchtext.vocab as vocab

from util.constant import DATA_CACHE_DIR, GLOVE_DATA_DIR

res = vocab.pretrained_aliases.keys()
print(res)


def lookup_vocab(name):
    res = [key for key in vocab.pretrained_aliases.keys() if name in key]
    print(res)


lookup_vocab('glove')
lookup_vocab('fasttext')

# ['glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
# 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']


cache_dir = '{}/glove/'.format(DATA_CACHE_DIR)
# glove = vocab.pretrained_aliases["glove.6B.50d"](cache=cache_dir)
# glove = vocab.GloVe(name='6B', dim=50, cache=cache_dir)
glove = vocab.pretrained_aliases["glove.6B.50d"](cache=GLOVE_DATA_DIR)

print("一共包含%d个词" % len(glove.stoi))

print(glove.stoi['beautiful'])
print(glove.itos[3366])


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
    print("")


# 求类比词 求类比词问题可以定义为：对于类比关系中的4个词 a:b::c:da:b::c:d，给定前3个词aa、bb和cc，求dd。设词ww的词向量为vec(w)vec(w)。求类比词的思路是，搜索与vec(c)+vec(
# b)−vec(a)vec(c)+vec(b)−vec(a)的结果向量最相似的词向量。
def get_analogy(token_a, token_b, token_c, embed):
    vecs = [embed.vectors[embed.stoi[t]] for t in [token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.vectors, x, 1)
    return embed.itos[topk[0]]


if __name__ == '__main__':
    get_similar_tokens('chip', 3, glove)
    get_similar_tokens('baby', 3, glove)
    get_similar_tokens('beautiful', 3, glove)

    res = get_analogy('man', 'woman', 'son', glove)  # 'daughter'
    print(res)
    res = get_analogy('beijing', 'china', 'tokyo', glove)  # 'japan'
    print(res)
    res = get_analogy('bad', 'worst', 'big', glove)  # 'biggest'
    print(res)
    res = get_analogy('do', 'did', 'go', glove)  # 'went'
    print(res)

    pass
