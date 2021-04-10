#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : cut_words.py
# @Author: sl
# @Date  : 2021/4/10 -  下午12:25

"""
分词学习
- 正向最大匹配
- 逆向最大匹配
- 双向最大匹配
- 统计分词 HMM

"""

import glob
import os
import random
import re
from datetime import datetime, timedelta

import jieba
from dateutil.parser import parse

from util.common_utils import get_TF
from util.file_utils import get_news_path, get_content
from util.logger_utils import get_log
import os

from util.nlp_utils import stop_words

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))


# 正向最大匹配
class MM(object):

    def __init__(self, window_size=3, word_dict=None):
        self.window_size = window_size
        self.word_dict = word_dict
        self.init()

    def init(self):
        if self.word_dict is None:
            self.word_dict = ['研究', '研究生', '生命', '命', '的', '起源']

    def cut(self, text):
        result = []
        index = 0
        text_length = len(text)
        while text_length > index:
            for size in range(self.window_size + index, index, -1):
                piece = text[index:size]
                if piece in self.word_dict:
                    index = size - 1
                    break
            index = index + 1
            result.append(piece)
        return result


# 逆向最大匹配
class IMM(object):

    def __init__(self, dic_path):
        self.word_dict = set()
        self.window_size = 0

        self.init(dic_path)

    def init(self, path):
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                self.word_dict.add(line)
                if len(line) > self.window_size:
                    self.window_size = len(line)

    def cut(self, text):
        result = []
        index = len(text)
        while index > 0:
            word = None
            for size in range(self.window_size , 0, -1):
                if index - size < 0:
                    continue

                piece = text[(index - size):index]
                if piece in self.word_dict:
                    word = piece
                    result.append(word)
                    index -= size
                    break
            if word is None:
                index -= 1

        return result[::-1]

if __name__ == '__main__':
    # text = '研究生命的起源'
    # tokenizer = MM()

    text = '南京市长江大桥'
    tokenizer = IMM("../../data/nlp/imm_dic.utf8")
    result = tokenizer.cut(text)
    print('/ '.join(result))
    pass
