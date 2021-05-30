#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : qyer_utils.py
# @Author: sl
# @Date  : 2021/5/30 -  上午11:21
import os
import time

from basis.utils.random_user_agent import RandomUserAgentMiddleware
from util.constant import DATA_HTML_DIR, DATA_QUESTION_DIR
from util.file_utils import save_to_text, read_to_text, check_file_exists
from util.logger_utils import get_log
from util.nlp_utils import sent2features, extend_maps, process_data_for_lstmcrf
import requests
from urllib import parse

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))

SPECIAL_LIST_URL = "https://ask.qyer.com/api/special/list"
SPECIAL_QLIST_URL = "https://ask.qyer.com/api/special/qlist"
PAGE_SIZE = 10
QA_DELIMITER = "|"

def build_header(url):
    headers = {
        "authority": "ask.qyer.com",
        "cache-control": "max-age=0",
        "user-agent": RandomUserAgentMiddleware().get_user_agent(),
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "cookie": "_qyeruid=CgIBGWCxypZC80SrB7N6Ag==; _guid=Rfea04f6-bfb0-c1df-4b3a-cd3b55064bcd; new_uv=1; new_session=1; filters_tags=; PHPSESSID=78d15286a01316e836c96b751d4f39c0; source_url=https%3A//www.google.com.hk/; isnew=1622264485790; __utmc=253397513; __utmz=253397513.1622264497.1.1.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); ql_guid=QLd014d0-f488-424a-976f-92e5861feaeb; __utma=253397513.989199164.1622264497.1622264497.1622269721.2; ql_created_session=1; ql_stt=1622269720745; ql_vts=2; __utmb=253397513.8.10.1622269721; ql_seq=8",
        "sec-fetch-site": "same-origin",
        "sec-fetch-mode": "navigate",
        "sec-fetch-user": "?1",
        "sec-fetch-dest": "document",
        "referer": url
    }

    return headers


def send_http_request(url, data=None, method='GET'):
    """发送http 请求"""
    log.info("开始发送请求：{} - {} : {}".format(method, url,data))
    if method.upper() == 'GET':
        r = requests.get(url, headers=build_header(url)).content.decode("utf-8")
    elif method.upper() == 'POST':
        if data is not None:
            r = requests.post(url, data=data, headers=build_header(url)).content.decode("utf-8")
        else:
            r = requests.post(url, headers=build_header(url)).content.decode("utf-8")
    else:
        r = ""

    log.info("请求的返回结果：{}".format(r))
    return r


def get_special_list(page=1):
    """获取问答专题"""
    r = send_http_request(url=SPECIAL_LIST_URL, method='POST', data={"page": page})
    return r


def get_special_one_page(id,page=1):
    """获取问答专题 的具体内容"""
    r = send_http_request(url=SPECIAL_QLIST_URL, method='POST', data={"id": id,"page": page})
    # time.sleep(0.1)
    return r

def filter_text(text, filter_list=None):
    """过滤文本中的内容"""
    if filter_list is None:
        filter_list = [" ", "​", " ", "\r", "\n", QA_DELIMITER]
    result = str(text).rstrip()
    for name in filter_list:
        target = ""
        if "\n" == name:
            target = "。"
        result = str(result).replace(name, target)

    return result


def build_question_filename(keyword, total):
    """构建问答路径"""
    filename = "{}/{}_{}.txt".format(DATA_QUESTION_DIR, keyword, total)
    return filename

def save_question(question_list, keyword, total):
    """保存问题"""
    filename = build_question_filename(keyword, total)

    contents = "\n".join(question_list)
    contents += "\n"
    save_to_text(filename, contents, 'a')
    log.info("保存问题：关键字：{} 总共：{} 条，本次保存：{} 条".format(keyword, total, len(question_list)))



if __name__ == '__main__':
    pass
