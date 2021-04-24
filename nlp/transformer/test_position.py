#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_position.py
# @Author: sl
# @Date  : 2021/4/24 -  下午4:15

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


def get_positional_encoding(max_seq_len, embed_dim):
    # 初始化一个positional encoding
    # embed_dim: 字嵌入的维度
    # max_seq_len: 最大的序列长度
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
        if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)])

    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # dim 2i 偶数
    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim 2i+1 奇数
    return positional_encoding


def show_posision1():
    positional_encoding = get_positional_encoding(max_seq_len=100, embed_dim=16)
    plt.figure(figsize=(10, 10))
    sns.heatmap(positional_encoding)
    plt.title("Sinusoidal Function")
    plt.xlabel("hidden dimension")
    plt.ylabel("sequence length")

if __name__ == '__main__':

    show_posision1()
    pass