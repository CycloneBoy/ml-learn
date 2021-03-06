#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : bilibili_course_spider.py
# @Author: sl
# @Date  : 2021/5/22 -  下午8:47

import os
import queue
import threading
import time
import urllib.request
import re

import sys
from you_get import common as you_get

import sys
from you_get import common as you_get
from multiprocessing import Pool

from util.constant import BILIBILI_VIDEO_IMAGE_DIR
from util.file_utils import get_file_name_list
from util.logger_utils import get_log

log = get_log("{}.log".format(os.path.basename(__file__)[:-3]))


# 获取所有需要下载的url
def get_urls(base_url, start_index, end_index):
    urls = []
    for i in range(start_index, end_index):
        url = base_url + str(i)
        urls.append(url)
    return urls


def download(urls, directory=BILIBILI_VIDEO_IMAGE_DIR):
    log.info("开始下载:{}".format(urls))
    sys.argv = ['you-get', '-o', directory, '--no-caption', urls,'--debug']
    you_get.main()
    log.info("完成下载:{}".format(urls))


def get_need_download(path=BILIBILI_VIDEO_IMAGE_DIR, start_index=59, end_index=129):
    file_list = get_file_name_list(path, "*.mp4")
    exist_list = []
    for name in file_list:
        index = re.findall(r"P(.+?)\.", name)
        if len(index) > 0:
            exist_list.append(int(index[0]))
        # log.info("{}".format(index))

    need_list = [i for i in range(start_index, end_index + 1) if i not in set(exist_list)]

    log.info("已经下载:{} , 还需要下载:{}".format(len(exist_list), len(need_list)))
    log.info("需要下载列表:{}".format(", ".join([ str(i) for i in need_list])))


    return need_list


if __name__ == '__main__':

    url = 'https://www.bilibili.com/video/BV1Zt4y1a7Xr?from=search&seid=755424765684966116'
    url_t = "https://www.bilibili.com/video/BV1Zt4y1a7Xr?p="
    url_t = "https://www.bilibili.com/video/BV1Nv41177cA?p="

    # sys.argv = ['you-get', '-o', BILIBILI_VIDEO_IMAGE_DIR, url, '-l']
    # you_get.main()

    # urls = get_urls(url_t,88,161)
    # pool = Pool(10)
    # pool.map(download, urls)
    # pool.close()
    # pool.join()

    #
    # log.info("{}".format("test."))
    # log.debug("{}".format("test."))
    # name = "/home/sl/data/bilibili/京东nlp训练营 (P130. Transformer的代码实现-3).mp4"
    # index = re.findall(r"P(.+?)\.", name)
    # log.info("提取:{}".format(index))

    need_list = get_need_download(start_index=55, end_index=128)
    for i in need_list:
        log.info("{}".format(i))
        url = url_t + str(i)
        download(url)
    pass
