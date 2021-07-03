#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : nlp_utils.py
# @Author: sl
# @Date  : 2021/4/10 -  上午11:45


"""
NLP 相关的工具类
"""
import pickle

import jieba
import jieba.posseg as psg
import opencc

import re
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

from util.constant import DATA_TXT_STOP_WORDS_DIR, TEST_SUMMARY_1


def stop_words(path=DATA_TXT_STOP_WORDS_DIR):
    """
    获取停用词
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        return [l.strip() for l in f]


def seg_to_list(sentence, pos=False):
    """
    分词方法，调用结巴接口
    :param sentence: 
    :param pos: 
    :return: 
    """
    if not pos:
        # 不进行词性标注的分词方法
        seg_list = jieba.cut(sentence)
    else:
        # 进行词性标注的分词方法
        seg_list = psg.cut(sentence)
    return seg_list


def merge_maps(dict1, dict2):
    """
    用于合并两个word2id或者两个tag2id
    :param dict1:
    :param dict2:
    :return:
    """
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1


def save_model(model, file_name):
    """
    用于保存模型
    :param model:
    :param file_name:
    :return:
    """
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)


def load_model(file_name):
    """
    用于加载模型
    :param file_name:
    :return:
    """
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
    return model


def extend_maps(word2id, tag2id, for_crf=True):
    """
    LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    :param word2id:
    :param tag2id:
    :param for_crf:
    :return:
    """
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id


def process_data_for_lstmcrf(word_lists, tag_lists, test=False):
    """
    lstmcrf 特殊处理
    :param word_lists:
    :param tag_lists:
    :param test:
    :return:
    """
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append('<end>')
        # 如果是测试数据，就不需要加end token了
        if not test:
            tag_lists[i].append('<end>')

    return word_lists, tag_lists


def flatten_lists(lists):
    """
    合并两个list
    :param lists:
    :return:
    """
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list


# ******** CRF 工具函数*************
def word2features(sent, i):
    """
    抽取单个字的特征
    :param sent:
    :param i:
    :return:
    """
    word = sent[i]
    prev_word = "<s>" if i == 0 else sent[i - 1]
    next_word = "</s" if i == (len(sent) - 1) else sent[i + 1]
    # 使用的特征：
    # 前一个词，当前词，后一个词，
    # 前一个词+当前词， 当前词+后一个词
    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word + word,
        'w:w+1': word + next_word,
        'bias': 1
    }
    return features


def sent2features(sent):
    """
    抽取序列特征
    :param sent:
    :return:
    """
    return [word2features(sent, i) for i in range(len(sent))]


def zh_convert(sentence, t2s=True):
    """繁体转简体"""
    converter = opencc.OpenCC('t2s.json')
    res = converter.convert(sentence)
    return res


def cut_sentence(sentence):
    """
    分句
    :param sentence:
    :return:
    """
    match = '[.。？！?!\n\r]'
    match1 = '[，:;!?。：；？！\n\r]'
    res_sen = re.compile(match)  # .不加是因为不确定.是小数还是英文句号(中文省略号......)
    sentences = res_sen.split(sentence)
    sen_cuts = []
    for sen in sentences:
        if sen and str(sen).strip():
            sen_cuts.append(sen)
    return sen_cuts


def extract_chinese(text):
    """
     只提取出中文、字母和数字
    :param text:
    :return:
    """
    chinese_extract = ''.join(re.findall(u"([\u4e00-\u9fa5A-Za-z0-9@. ])", text))
    return chinese_extract


def remove_urls(text):
    """
     删除https/http等无用url
    :param text:
    :return:
    """
    text_remove_url = re.sub(r'(全文：)?(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',
                             '', text, flags=re.MULTILINE)
    return text_remove_url


def gram_uni_bi_tri(text):
    """
      获取文本的unigram, trugram, bigram等特征
    :param text:
    :return:
    """
    len_text = len(text)
    gram_uni = []
    gram_bi = []
    gram_tri = []
    for i in range(len_text):
        if i + 3 <= len_text:
            gram_uni.append(text[i])
            gram_bi.append(text[i:i + 2])
            gram_tri.append(text[i:i + 3])
        elif i + 2 <= len_text:
            gram_uni.append(text[i])
            gram_bi.append(text[i:i + 2])
        elif i + 1 <= len_text:
            gram_uni.append(text[i])
        else:
            break
    return gram_uni, gram_bi, gram_tri


def tfidf_fit(sentences):
    """ tfidf相似度"""

    model = TfidfVectorizer(ngram_range=(1, 2),
                            stop_words=stop_words(),
                            max_features=10000,
                            token_pattern=r"(?u)\b\w+\b",  # 过滤停用词
                            min_df=1, max_df=0.9,
                            use_idf=True, smooth_idf=True, sublinear_tf=True)
    matrix = model.fit_transform(sentences)
    return matrix


if __name__ == '__main__':
    # words = stop_words()
    # print("停用词数量:%d" % len(words))
    # print('/ '.join(words[1000:1200]))
    line = 'CNN在图像领域的成功也表明，深度卷积神经网络能够提取更加复杂和高级的特征，尤其是深度残差网络(ResNet)等的流行。那么，深度卷积神经网络在自然语言处理NLP领域，究竟有没有优势呢?'
    res = list(jieba.cut(line))
    print(list(jieba.cut(line)))
    print(list(jieba.cut(line, cut_all=False, HMM=True)))
    print(list(jieba.cut(line, cut_all=False, HMM=False)))

    line2 = TEST_SUMMARY_1[1]
    print(cut_sentence(line))
    print(cut_sentence(line2))

    grams = gram_uni_bi_tri(line2)

    print(grams[0])
    print(grams[1])
    print(grams[2])

    pass
