#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : process_data.py
# @Author: sl
# @Date  : 2021/5/30 -  下午8:50

"""
处理问答
"""

import os
import time

from basis.utils.random_user_agent import RandomUserAgentMiddleware
from nlp.question.data.model import QaModel
from util.constant import DATA_HTML_DIR, DATA_QUESTION_DIR, DATA_CACHE_DIR
from util.file_utils import save_to_text, read_to_text, check_file_exists
from util.logger_utils import get_log
from util.nlp_utils import sent2features, extend_maps, process_data_for_lstmcrf
import requests
from urllib import parse

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))


def read_data():
    """读取数据"""
    filename = os.path.join(DATA_CACHE_DIR, "question/travel_question_64102.txt")
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
            result_list.append(row)

    log.info("重复数量:{}".format(duplicate_number))

    filename = os.path.join(DATA_CACHE_DIR, "question/travel_question_filter_{}.txt".format(len(result_list)))
    save_to_text(filename,"\n".join(result_list))


if __name__ == '__main__':
    read_data()
    pass
