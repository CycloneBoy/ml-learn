#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: sl
# @Date  : 2021/8/27 - 下午4:49
import json
import os
import random
import re
import time
from datetime import timedelta

import jieba
import numpy as np
import torch
from hanziconv import HanziConv
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from transformers import BertTokenizer, XLNetTokenizer, DistilBertTokenizer

from nlp.match.bert.config import BERT_PATH

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def get_labels_from_list(label_type='bios'):
    "CLUENER TAGS"
    tag_list = ["出品公司", "国籍", "出生地", "民族", "出生日期", "毕业院校", "歌手", "所属专辑", "作词", "作曲", "连载网站", "作者", "出版社", "主演", "导演",
                "编剧", "上映时间", "成立日", ]
    return tag_list


def load_tag_from_file(path):
    result = []
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            result.append(line.strip())
    return result


def load_tag_from_json(path):
    result = []
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
        for k, v in data.items():
            result.append(v)
    return result


def load_tag():
    tags = ["0", "1"]
    id2tag = {i: label for i, label in enumerate(tags)}
    tag2id = {label: i for i, label in enumerate(tags)}

    return tags, tag2id, id2tag


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average="macro"):
    acc = simple_accuracy(preds, labels)
    metrics = dict()
    metrics["f1"] = f1_score(labels, preds, average=average)
    metrics["precision"] = precision_score(labels, preds, average=average)
    metrics["acc"] = acc
    metrics["recall"] = recall_score(labels, preds, average=average)

    return metrics


def now_str(format="%Y-%m-%d_%H"):
    return time.strftime(format, time.localtime())


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def check_file_exists(filename, delete=False):
    """检查文件是否存在"""
    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print("文件夹不存在,创建目录:{}".format(dir_name))


def load_tokenizer(model_name_or_path=BERT_PATH, args=None):
    if "xlnet" == args.pretrained_model_name:
        tokenizer = XLNetTokenizer.from_pretrained(model_name_or_path)
    elif "bert-distil" == args.pretrained_model_name:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name_or_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


def get_checkpoint_dir(file_dir="./"):
    checkpoints = []
    for root, dirs, files in os.walk(file_dir):
        checkpoints.extend(dirs)

    max_step = 0
    checkpoint_dir = None
    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else 0
        if int(global_step) > max_step:
            max_step = int(global_step)
            checkpoint_dir = checkpoint
    return checkpoint_dir, max_step


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(sequences_lengths)).type_as(sequences_lengths)
    # idx_range = sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, revese_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, revese_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


def get_mask(sequences_batch, sequences_lengths):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask


def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.
    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.
    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


''' 把句子按字分开，中文按字分，英文数字按空格, 大写转小写，繁体转简体'''


def get_char_list(query):
    query = HanziConv.toSimplified(query.strip())
    regEx = re.compile('[\\W]+')  # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r'([\u4e00-\u9fa5])')  # [\u4e00-\u9fa5]中文范围
    sentences = regEx.split(query.lower())
    str_list = []
    for sentence in sentences:
        if res.split(sentence) == None:
            str_list.append(sentence)
        else:
            ret = res.split(sentence)
            str_list.extend(ret)
    return [w for w in str_list if len(w.strip()) > 0]


def get_word_list(query):
    # 繁体转简体
    query = HanziConv.toSimplified(query.strip())
    # 大写转小写
    query = query.lower()
    # 利用jieba进行分词
    words = ' '.join(jieba.cut(query)).split(" ")
    return words


# 加载字典
def load_vocab(vocab_file):
    vocab = [line.strip() for line in open(vocab_file, encoding='utf-8').readlines()]
    if PAD_TOKEN not in set(vocab):
        vocab.insert(0, PAD_TOKEN)
    if UNK_TOKEN not in set(vocab):
        vocab.insert(1, UNK_TOKEN)
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word, vocab


# word->index
def word_index(p_sentences, h_sentences, word2idx, max_seq_len):
    p_list, p_length, h_list, h_length = [], [], [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word] if word in word2idx.keys() else 1 for word in p_sentence]  # 1为[UNK]对应的索引
        h = [word2idx[word] if word in word2idx.keys() else 1 for word in h_sentence]
        p_list.append(p)
        p_length.append(min(len(p), max_seq_len))
        h_list.append(h)
        h_length.append(min(len(h), max_seq_len))
    # p_list = pad_sequences(p_list, maxlen=max_seq_len)
    # h_list = pad_sequences(h_list, maxlen=max_seq_len)
    return p_list, p_length, h_list, h_length


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences
    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。
    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值
    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)  # 对应[PAD]为0
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def pad_seq(seq, max_len=64, value=0):
    """
    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度
    """
    x = [value] * max_len
    trunc = seq[:max_len]
    x[:len(trunc)] = trunc

    return x


if __name__ == '__main__':
    tags, tag2id, id2tag = load_tag()
    print(tags)
    print(tag2id)
    print(id2tag)

    line1 = "不满足微众银行条件,“您未满足微众银行审批要求，无法查看额度”，这是为什么？什么原因呢,1"
    res = get_char_list(line1)
    print(res)
    res = get_word_list(line1)
    print(res)

    seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ]
    # res = pad_sequences(seq, max_len=15)
    print(res)
