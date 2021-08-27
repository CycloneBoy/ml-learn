#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: sl
# @Date  : 2021/8/27 - 下午4:49
import json
import os
import time
from datetime import timedelta

from nlp.re.casrel.config import RELATION_DATA_DIR
from util.logger_utils import Logger


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


def load_tag(path=RELATION_DATA_DIR, label_type='bios'):
    if path is not None:
        # tags = load_tag_from_file(path)
        tags = load_tag_from_json(path)
    else:
        tags = get_labels_from_list(label_type)

    id2tag = {i: label for i, label in enumerate(tags)}
    tag2id = {label: i for i, label in enumerate(tags)}

    return tags, tag2id, id2tag


def now_str(format="%Y-%m-%d_%H"):
    return time.strftime(format, time.localtime())


def init_logger(filename='test'):
    log = Logger('{}_{}.log'.format(filename, now_str()), level='debug')
    return log.logger


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


logger = init_logger("./log/test")

if __name__ == '__main__':
    tags, tag2id, id2tag = load_tag()
    print(tags)
    print(tag2id)
    print(id2tag)
