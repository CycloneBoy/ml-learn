#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : qyer_special.py
# @Author: sl
# @Date  : 2021/5/30 -  上午11:23
import json
import os

from requests.utils import get_encodings_from_content

from basis.spider.travel.qyer_utils import send_http_request, get_special_list, get_special_one_page, PAGE_SIZE, \
    filter_text, QA_DELIMITER, save_question
from basis.utils.random_user_agent import RandomUserAgentMiddleware
from util.constant import DATA_HTML_DIR, DATA_QUESTION_DIR, DATA_JSON_DIR
from util.file_utils import save_to_text, read_to_text, check_file_exists, save_to_json, load_to_json
from util.logger_utils import get_log
from util.nlp_utils import sent2features, extend_maps, process_data_for_lstmcrf
import requests
from urllib import parse

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))


def parse_result(res):
    """解析结果"""
    result = []
    res_json = json.loads(res)
    if "data" in res_json:
        for data in res_json["data"]:
            # log.info("解析：{}".format(data))
            result.append(data)

    return result


def get_special_data(max_page=5):
    """获取问答特殊专题"""
    result = []
    for i in range(1, max_page):
        res = get_special_list(i)
        res_list = parse_result(res)
        result.extend(res_list)

    filename = "{}/special_list.json".format(DATA_JSON_DIR)
    save_to_json(filename, result)
    return result


def get_special_page(id, total):
    """获取一个专题问答的问答"""

    total_page = int(total / PAGE_SIZE) + 1

    for page in range(1, total_page):
        res = get_special_one_page(id, page)
        res_list = parse_result(res)

        result_list = []
        for res_one in res_list:
            number = filter_text(res_one['qid'])
            ask_item_question = filter_text(res_one['title'])
            ask_item_answer = filter_text(res_one['content'])
            ask_item_time = filter_text(res_one['addtime'])
            result = [str(number), ask_item_question, ask_item_answer, ask_item_time]
            result_list.append(QA_DELIMITER.join(result))

        save_question(result_list, "问答专题", "1000")

        # result.extend(result_list)


if __name__ == '__main__':
    # special_list = get_special_data()

    filename = "{}/special_list.json".format(DATA_JSON_DIR)
    special_list = load_to_json(filename)

    for index, one_row in enumerate(special_list):
        if index <= 34:
            continue
        log.info("抓取：{}".format(index + 1))
        url = one_row["url"]
        id = url[url.rfind("/") + 1: -5]
        total = one_row["question_cnt"]

        get_special_page(id, total)
    pass
