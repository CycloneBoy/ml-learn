#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : qyer_question.py
# @Author: sl
# @Date  : 2021/5/29 -  下午2:30
import re

from bs4 import BeautifulSoup
from lxml import etree

from basis.spider.travel.qyer_utils import build_header, QA_DELIMITER, PAGE_SIZE, filter_text, save_question
from basis.utils.random_user_agent import RandomUserAgentMiddleware

import traceback
import os

from util.constant import DATA_HTML_DIR, DATA_QUESTION_DIR
from util.file_utils import save_to_text, read_to_text, check_file_exists
from util.logger_utils import get_log
from util.nlp_utils import sent2features, extend_maps, process_data_for_lstmcrf
import requests
from urllib import parse

from util.time_utils import now_str

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))

BASE_URL = "https://ask.qyer.com/search?q={}&filter_time=0&page={}"



def build_filename(keyword, page):
    """构建文件路径"""
    filename = "{}/{}/{}_{}.html".format(DATA_HTML_DIR, keyword, keyword, page)
    return filename





def search_question(keyword="东南亚", page="1"):
    keyword_quote = parse.quote(keyword)
    url = BASE_URL.format(keyword_quote, page)
    log.info("url:{}".format(url))

    html = requests.get(url, headers=build_header(url)).content.decode("utf-8")

    filename = build_filename(keyword, page)
    save_to_text(filename, html)

    return html


def get_total(total):
    """提取总数"""
    return int(total[total.index("共") + 2: total.index("条")])


def get_time(time_text):
    """提取时间"""

    try:
        ask_item_time_match = re.search(r"(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2})", time_text)
        if len(ask_item_time_match.groups()) > 0:
            return  ask_item_time_match.group(0)
        else:
            return now_str(format="%Y-%m-%d %H:%M")
    except Exception:
        # raise ValueError("错误是日期格式或日期,格式是年-月-日")
        return now_str(format="%Y-%m-%d %H:%M")




def extract_question(keyword, page):
    """解析问答"""
    filename = build_filename(keyword, page)
    html_str = read_to_text(filename)

    return extract_question_html(html_str, page)


def extract_question_html(html_str, page):
    """解析html中的问答"""
    html = BeautifulSoup(html_str, "lxml")

    question_total = html.select("div.ask__search__total > span")[0].string
    total = get_total(question_total)
    log.info("问题总数：{}".format(total))

    ask_item_list = html.select("div.ask_item_main_item_list_content")

    result_list = []

    for index, ask_item in enumerate(ask_item_list):
        ask_item_question = filter_text(ask_item.select("h4.ask_item_main_item_list_title > a")[0].get_text())
        ask_item_answer = filter_text(ask_item.select("blockquote.qyer_spam_text_filter")[0].get_text())
        ask_item_time_str = ask_item.select("div.add_headPort")[0].get_text()

        ask_item_time = get_time(ask_item_time_str)

        log.info("{}".format(""))
        log.info("{}".format("-" * 50))
        number = (int(page) - 1) * PAGE_SIZE + index + 1
        log.info("id: {} 第 {} 个问题:{}".format(number, index, ask_item_question))
        log.info("id: {} 第 {} 个答案:{}".format(number, index, ask_item_answer))
        log.info("id: {} 第 {} 个时间:{}".format(number, index, ask_item_time))
        # log.info("data:{}".format(ask_item))

        log.info("{}".format("-" * 50))

        result = [str(number), ask_item_question, ask_item_answer, ask_item_time]
        result_list.append(QA_DELIMITER.join(result))

    return total, result_list


def get_keyword_total_question(keyword, page=1):
    """获取关键字的总的问题数"""
    html = search_question(keyword, page)

    question_total, result_list = extract_question_html(html, page)
    log.info("获取关键字：{} 的总的问题数：{}".format(keyword, question_total))
    return question_total


def crawl_question_page(keyword="", total=0, begin_page=1, end_page=1):
    """爬取问答"""

    total_page = int(total / PAGE_SIZE) + 1
    if total == 0:
        total_page = 1
    if end_page == 1:
        end_page = total_page

    error_list = []

    for page in range(begin_page, end_page):
        try:
            log.info("开始爬取关键字： {} 的第：{} 页".format(keyword, page))
            html = search_question(keyword, page)

            question_total, result_list = extract_question_html(html, page)

            save_question(result_list, keyword, question_total)
            log.info("结束爬取关键字： {} 的第：{} 页".format(keyword, page))
        except Exception as e:
            log.error("爬取出错：关键字： {} 的第：{} 页".format(keyword, page))
            error_list.append(page)
            traceback.print_exception(e)

    for name in error_list:
        log.info("本轮爬取关键字：{} 出错的页码：{}".format(keyword, name))


def crawl_question_keyword_dict(keyword_dict):
    """爬取指定关键字信息"""

    for key, value in keyword_dict.items():
        question_total = get_keyword_total_question(key)
        log.info("开始爬取关键字：{} ，设定总量：{} ->实际总量 {}".format(key, value, question_total))

        crawl_question_page(key, total=question_total)


def crawl_question_keyword_list(keyword_list):
    """爬取指定关键字信息"""

    for key in keyword_list:
        question_total = get_keyword_total_question(key)
        log.info("开始爬取关键字：{} ，实际总量: {}".format(key, question_total))

        crawl_question_page(key, total=question_total)


if __name__ == '__main__':
    keyword = "东南亚"
    total = 2000

    # crawl_question_page(keyword, total)
    # crawl_question_page(keyword, begin_page=21, end_page=200)

    keyword_dict = {
        "跳岛游": 1130,
        "泰国": 2000
    }

    # crawl_list = ["武汉", "荷兰", "里斯本", "过境签", "瑞士", "青海湖", "伊朗", "韩国", "意大利", "苏州", "好玩", "巴黎", "黑山", "沙巴", "马德里", "南非",
    #               "尼斯", "圣彼得堡", "吉隆坡", "成都", "推荐", "但是", "布拉格", "攻略", "智利", "花莲", "坐车", "飞机", "英国", "房车", "租赁", ]

    crawl_list = ["成都", "推荐", "但是", "布拉格", "攻略", "智利", "花莲", "坐车", "飞机", "英国", "房车", "租赁", ]

    crawl_question_keyword_list(crawl_list)
    pass
