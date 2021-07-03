#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : keword_jieba.py
# @Author: sl
# @Date  : 2021/7/3 -  上午11:01

# 适配linux
import sys
import os
# path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) # 获取上级目录
from util.constant import TEST_SUMMARY_1

path_root = os.path.abspath(os.path.join(os.getcwd(), "../..")) # 获取上上级目录
sys.path.append(path_root)

import jieba.analyse as analyse
import logging
import jieba



def keyword_tfidf(sentence,rate=1.):
    """
     使用tf-dif获取关键词构建
    :param sentence:  str, input sentence
    :param rate:  float, 0-1
    :return: str
    """
    sen_words = jieba._lcut(sentence)
    top_k = int(len(sen_words) * rate)

    keyword = analyse.extract_tags(sentence,topK=top_k,withWeight=False,
                                   withFlag=False)
    keyword_sort = [k if k in keyword else '' for k in sen_words]
    return ''.join(keyword_sort)

def keyword_textrank(sentence,rate=1.,
                     allow_pos=('an', 'i', 'j', 'l', 'r', 't',
                                         'n', 'nr', 'ns', 'nt', 'nz',
                                         'v', 'vd', 'vn')):
    """
        使用text-rank获取关键词构建
    :param sentence:  str, input sentence, 例: '大漠帝国是谁呀，你知道吗'
    :param rate: float, 0-1 , 例: '0.6'
    :param allow_pos: list, 例: ('ns', 'n', 'vn', 'v')
    :return: str, 例: '大漠帝国'
    """
    sen_words = jieba._lcut(sentence)
    top_k = int(len(sen_words) * rate)

    keyword = analyse.textrank(sentence, topK=top_k,allowPOS=allow_pos,
                               withWeight=False,withFlag=False)
    keyword_sort = [k if k in keyword else '' for k in sen_words]
    return ''.join(keyword_sort)




if __name__ == '__main__':
    sen = TEST_SUMMARY_1[1]

    sen_tf = keyword_tfidf(sentence=sen,rate=0.1)
    sen_rank = keyword_textrank(sentence=sen,rate=0.6)
    print(sen)
    print(sen_tf)
    print(sen_rank)
    pass