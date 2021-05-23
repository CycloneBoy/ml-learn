#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils_fasttext.py
# @Author: sl
# @Date  : 2021/5/11 -  下午11:07
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

# THUCNews 路径
from util.constant import DATA_EMBEDDING_SOGOU_CHAR

DATA_THUCNEWS_DIR = '/home/sl/workspace/python/a2020/ml-learn/data/nlp/THUCNews'

# 词表长度限制
MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, use_word, is_pretain=False):
    vocab = None

    def init():
        if use_word:
            tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        else:
            tokenizer = lambda x: [y for y in x]  # char-level
        if os.path.exists(config.vocab_path):
            vocab = pkl.load(open(config.vocab_path, 'rb'))
        else:
            vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
            pkl.dump(vocab, open(config.vocab_path, 'wb'))
        print(f"Vocab size: {len(vocab)}")
        return vocab

    # 初始化
    if not is_pretain:
        vocab = init()

    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def _load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                # fasttext ngram
                buckets = config.n_gram_vocab
                bigram = []
                trigram = []

                # ------ngram------
                for i in range(pad_size):
                    bigram.append(biGramHash(words_line, i, buckets))
                    trigram.append(triGramHash(words_line, i, buckets))
                    # -----------------
                contents.append((words_line, int(label), seq_len, bigram, trigram))
        return contents  # [([...], 0), ([...], 1), ...]

    def load_dataset(path, pad_size=32, is_pretain=is_pretain):
        pass

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # xx = [xxx[2] for xxx in datas]
        # indexx = np.argsort(xx)[::-1]
        # datas = np.array(datas)[indexx]
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, bigram, trigram), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batchs = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batchs = self._to_tensor(batchs)
            return batchs
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batchs = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batchs = self._to_tensor(batchs)
            return batchs

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = os.path.join(DATA_THUCNEWS_DIR, "data/train.txt")
    vocab_dir = os.path.join(DATA_THUCNEWS_DIR, "data/vocab.pkl")
    # pretrain_dir = os.path.join(DATA_THUCNEWS_DIR, "data/sgns.sogou.char")
    pretrain_dir = DATA_EMBEDDING_SOGOU_CHAR
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, 'r', encoding='utf-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.array(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
