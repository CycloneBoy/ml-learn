#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : nlp_utils.py
# @Author: sl
# @Date  : 2021/4/10 -  上午11:45


"""
NLP 相关的工具类
"""
from util.constant import DATA_TXT_STOP_WORDS_DIR


def stop_words(path=DATA_TXT_STOP_WORDS_DIR):
    """
    获取停用词
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        return [l.strip() for l in f]

if __name__ == '__main__':
    words = stop_words()
    print("停用词数量:%d" % len(words))
    print('/ '.join(words[1000:1200]))
    pass
