#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : file_utils.py
# @Author: sl
# @Date  : 2021/4/10 -  上午11:24

'''
文件处理的工具类

'''
from util.constant import DATA_TXT_NEWS_DIR, DATA_TXT_STOP_WORDS_GITHUB_DIR, BILIBILI_VIDEO_IMAGE_DIR, DATA_HTML_DIR, \
    DATA_QUESTION_ANSWER_DIR, DATA_CACHE_DIR
from util.logger_utils import get_log
import os
import glob
import random
import json

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


def save_to_text(filename, content, mode='w'):
    """
    保存为文本
    :param filename:
    :param content:
    :return:
    """
    check_file_exists(filename)
    with open(filename, mode, encoding='utf-8') as f:
        f.writelines(content)


def save_to_json(filename, content):
    """
    保存map 数据
    :param filename:
    :param maps:
    :return:
    """
    check_file_exists(filename)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False)


def load_to_json(filename):
    """
    加载 数据
    :param filename:
    :param maps:
    :return:
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_file_name_list(path, type="*.txt"):
    """获取指定路径下的指定类型的所有文件"""
    files = glob.glob(os.path.join(path, type))
    return files


def check_file_exists(filename, delete=False):
    """检查文件是否存在"""
    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        log.info("文件夹不存在,创建目录:{}".format(dir_name))


def read_to_text(path, encoding='utf-8'):
    """读取txt 文件"""
    with open(path, 'r', encoding=encoding) as f:
        content = f.read()
        return content


def list_file(file_dir, endswith=""):
    """读取文件列表"""
    file_list = []
    for file in os.listdir(file_dir):
        if file.endswith(endswith):
            file_list.append(file)

    return file_list


def build_qa_dataset(file_dir):
    """读取问答数据集"""
    file_list = list_file(file_dir, ".txt")

    filename = os.path.join(DATA_CACHE_DIR, "question/travel_question_63752.txt")
    total = 0
    for file in file_list:
        total_line = int(file[file.find("_") + 1:file.find(".")])
        total += total_line
        path = os.path.join(file_dir, file)
        contents = read_to_text(path)
        save_to_text(filename, contents, 'a')

    log.info("文件数：{}，总共问题数量：{}".format(len(file_list), total))


if __name__ == '__main__':
    # test_get_one_news()

    # build_stop_words()

    filename = "../data/test/test1.txt"
    # save_to_text(filename, 'hhhhhhhhhhhhhhhhhhhhhhhhhhhhh\nrrrr\n333333')

    # file_list = get_file_name_list(BILIBILI_VIDEO_IMAGE_DIR, "*.mp4")
    # for name in file_list:
    #     log.info("{}".format(name))

    filename = "{}/{}/{}_{}.html".format(DATA_HTML_DIR, "test", "test", 1)
    # check_file_exists(filename)

    build_qa_dataset(DATA_QUESTION_ANSWER_DIR)

    # file = "巴黎_2000.txt"
    # total_line = file[file.find("_") + 1:file.find(".")]
    # print(total_line)
    pass
