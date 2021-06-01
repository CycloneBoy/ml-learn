#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : process_data.py
# @Author: sl
# @Date  : 2021/5/30 -  下午8:50

"""
处理问答
"""

import os
import random
import time

import opencc

from basis.utils.random_user_agent import RandomUserAgentMiddleware
from nlp.question.data.model import QaModel
from util.constant import DATA_HTML_DIR, DATA_QUESTION_DIR, DATA_CACHE_DIR, TEST_QA_LIST_10
from util.file_utils import save_to_text, read_to_text, check_file_exists
from util.logger_utils import get_log
from util.nlp_utils import sent2features, extend_maps, process_data_for_lstmcrf
import requests
from urllib import parse

from util.time_utils import time_cost

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))

converter = opencc.OpenCC('t2s.json')


def clean_text(sentence):
    """清理数据"""

    # 简繁体转换
    res = converter.convert(sentence)
    return res


def read_data():
    """读取数据"""
    filename = os.path.join(DATA_CACHE_DIR, "question/travel_question_filter_63752.txt")
    dataset = read_to_text(filename)
    split_data = dataset.split("\n")
    result_list = []
    duplicate_number = 0
    question_set = set()

    for row in split_data:
        qa = QaModel(row)
        if qa.question in question_set:
            duplicate_number += 1
        elif qa.question is None:
            continue
        else:
            question_set.add(qa.question)
            result_list.append(clean_text(row))

    log.info("重复数量:{}".format(duplicate_number))

    filename = os.path.join(DATA_CACHE_DIR, "question/travel_question_filter_{}.txt".format(len(result_list)))
    log.info("保存名称:{}".format(filename))

    save_to_text(filename, "\n".join(result_list))


def read_question(is_test=False):
    """读取文本"""

    if is_test:
        # 加载测试数据
        log.info("加载测试数据")
        return TEST_QA_LIST_10
    filename = os.path.join(DATA_CACHE_DIR, "question/travel_question_filter_59719.txt")
    dataset = read_to_text(filename)
    split_data = dataset.split("\n")
    return split_data


@time_cost
def load_question(is_test=False):
    """加载数据"""
    split_data = read_question(is_test)
    result_list = []

    for row in split_data:
        qa = QaModel(row)
        result_list.append(qa)

    log.info("读取问答数据集:{}".format(len(result_list)))
    log.info("示例：")
    for i in range(0, 10):
        log.info("{}".format(result_list[random.randint(0, len(result_list) - 1)]))

    return result_list


if __name__ == '__main__':
    read_data()
    # qa_list = load_question(is_test=True)



    pass
