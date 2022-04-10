#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : extract_user_info.py
# @Author: sl
# @Date  : 2022/4/9 - 下午9:25
import re
import traceback

import execjs
import requests
from scrapy import Selector

from basis.spider.mafengwo.extract_base import ExtractSpiderBase
from util.logger_utils import logger
from util.v2.constants import Constants
from util.v2.file_utils import FileUtils


class ExtractUserInfo(ExtractSpiderBase):

    def __init__(self, url=None, use_driver=False, retry=False):
        super().__init__(url)
        self.use_driver = use_driver
        self.retry = retry

    def get_user_info(self, url=None, use_driver=False, retry=False):
        """
        获取用户详情

        :param url:  http://www.mafengwo.cn/travel-scenic-spot/mafengwo/12711.html
        :return:
        """
        # file_name = FileUtils.get_url_file_path(url)
        # content = FileUtils.get_content(file_name, encoding="utf-8")
        content = self.get_url_content(url, use_driver=use_driver, retry=retry)

        response = Selector(text=content)

        mdd = response.xpath('//*[@id="container"]/div[1]/div/div[2]/div[2]/div/span/a').extract_first()
        logger.info(f"mdd:{mdd}")

    def get_travel_list_info(self, url=None, use_driver=False, retry=False):
        """
        获取用户详情

        :param url:  http://www.mafengwo.cn/travel-scenic-spot/mafengwo/12711.html
        :return:
        """
        content = self.get_url_content(url, use_driver=use_driver, retry=retry)

        response = Selector(text=content)

        note_list_url = response.xpath(
            ".//ul[contains(@class,'month-panel')]/li[contains(@class,'_j_hover')]/span[contains(@class,'mark')]/a/@href").extract()
        # mdd = response.xpath('//*[@id="container"]/div[1]/div/div[2]/div[2]/div/span/a').extract_first()
        # logger.info(f"mdd:{mdd}")
        note_list = []
        for note in note_list_url:
            if str(note).startswith("/i"):
                logger.info(f"note:{note}")
                note_list.append(note)

        logger.info(f"total:{len(note_list)}")
        FileUtils.save_to_json(f"{Constants.SPIDER_MFW_TRAVEL_HOT_DIR}/note.json", note_list)


def demo_extract_user_info():
    """

     session:{'__jsl_clearance': '1649515062.128|-1|8yLhZwotZjGVJEWNY8s6lar5crs%3D', '__jsluid_h': 'a829a4d56ca3e5918c136669a70f7896'}

    :return:
    """
    # url = "http://www.mafengwo.cn/u/56186565.html"
    # url = "http://www.mafengwo.cn/app/calendar.php"
    url = "http://www.mafengwo.cn/i/21448504.html"
    user_spider = ExtractUserInfo(url=url)

    # user_spider.get_user_info(url=url, use_driver=False)
    user_spider.get_travel_list_info(url=url, use_driver=True)


if __name__ == '__main__':
    pass
    demo_extract_user_info()
    # demo_get_user_html()
