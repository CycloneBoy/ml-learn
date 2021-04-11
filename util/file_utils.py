#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : file_utils.py
# @Author: sl
# @Date  : 2021/4/10 -  上午11:24

'''
文件处理的工具类

'''
from util.constant import DATA_TXT_NEWS_DIR, DATA_TXT_STOP_WORDS_GITHUB_DIR
from util.logger_utils import get_log
import os
import glob
import random

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))


def get_content(path, encoding='gbk'):
    """
    读取文本内容
    :param path:
    :param encoding:
    :return:
    """
    with open(path, 'r', encoding=encoding, errors='ignore') as f:
        content = ''
        for l in f:
            l = l.strip()
            content += l
        return content


def get_news_path(sub_path='C000008'):
    """
    获取新闻预料的路径
    :param sub_path:
    :return:
    """
    return os.path.join(DATA_TXT_NEWS_DIR, sub_path)


def build_stop_words(path=DATA_TXT_STOP_WORDS_GITHUB_DIR,
                     save_file_name=os.path.join(DATA_TXT_STOP_WORDS_GITHUB_DIR, 'stop_words.utf8')):
    files = glob.glob(os.path.join(path, '*.txt'))
    words = []
    for file in files:
        with open(file, 'r', errors='ignore') as f:
            for line in f.readlines():
                words.append(line)

    print("文件长度: %d 停用词长度:%d " % (len(files), len(words)))
    print("样例:%s " % words[100])

    result = []
    filter_set = set()
    for word in words:
        if word not in filter_set:
            result.append(word)
            filter_set.add(word)

    with open(save_file_name, 'w', errors='ignore') as f:
        f.writelines(result)
    print("保存处理后的停用词字典: %s ,停用词长度:%d " % (save_file_name, len(result)))


def test_get_one_news():
    files = glob.glob(os.path.join(get_news_path(), '*.txt'))
    corpus = [get_content(file) for file in files]

    sample_inx = random.randint(0, len(corpus))
    print(corpus[sample_inx])


def save_to_text(filename,content):
    """
    保存为文本
    :param filename:
    :param content:
    :return:
    """
    with open(filename,'w',encoding='utf-8') as f:
        f.writelines(content)

if __name__ == '__main__':
    # test_get_one_news()

    # build_stop_words()

    filename = "../data/test/test1.txt"
    save_to_text(filename,'hhhhhhhhhhhhhhhhhhhhhhhhhhhhh\nrrrr\n333333')

    pass
