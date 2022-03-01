#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : extract_city_info.py
# @Author: sl
# @Date  : 2022/2/18 - 下午10:20

from scrapy import Selector

import os
import time
import traceback
from collections import OrderedDict
from typing import List

from docx import Document
from docx.shared import Inches

from fpdf import FPDF
from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.wait import WebDriverWait

import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from basis.spider.mafengwo.extract_base import ExtractSpiderBase
from util.logger_utils import logger
from util.v2.constants import Constants
from util.v2.file_utils import FileUtils

"""
extract city info 
"""


def get_url_html(url):
    b = webdriver.Chrome()
    b.get(url)

    content = b.page_source
    # time.sleep(5)
    b.quit()

    FileUtils.save_to_text('./data.html', content)


class MafengwoSpider(ExtractSpiderBase):

    def __init__(self, url):
        super().__init__(url)

    def get_city_list(self, ):
        """
        get city list

        :param url:  http://www.mafengwo.cn/mdd/
        :return:
        """
        content = FileUtils.get_content(Constants.SPIDER_MAFENGWO_CITY_ALL_DIR, encoding="utf-8")

        response = Selector(text=content)

        el_hot_list = response.xpath("//a[contains(@href,'travel-scenic-spot')]").extract()

        hot_city_list = []
        for hot_city in el_hot_list:
            el_hot_city = Selector(text=hot_city)
            name = el_hot_city.xpath("//text()").extract_first()
            img_src = el_hot_city.xpath("//a/img/@src").extract_first()
            en_name = el_hot_city.xpath("//a/span/text()").extract_first()
            href = el_hot_city.xpath("//a/@href").extract_first()
            info = {
                "name": name,
                "en_name": en_name,
                "img_src": img_src,
                "href": href,
            }
            hot_city_list.append(info)

        logger.info(f'')
        FileUtils.save_to_json("./hot_city_list.json", hot_city_list)

    def get_city_info(self, url=None):
        """
        get city list

        :param url:  http://www.mafengwo.cn/travel-scenic-spot/mafengwo/12711.html
        :return:
        """
        content = FileUtils.get_content(Constants.SPIDER_MFW_CITY_YUNNAN_DIR, encoding="utf-8")

        response = Selector(text=content)

        mdd = response.xpath('//*[@id="container"]/div[1]/div/div[2]/div[2]/div/span/a').extract_first()

        city_global_info = self.extract_city_global_info(response)
        city_bar_list = self.extract_city_sub_bar_info(response)
        city_sales_list = self.extract_city_sales_list(response)
        city_overview_bar_list = self.extract_city_overview_bar_info(response)

        data = {
            "city_global_info": city_global_info,
            "city_bar_list": city_bar_list,
            "city_sales_list": city_sales_list,
            "city_overview_bar_list": city_overview_bar_list,
        }

        logger.info(f'{city_bar_list}')
        FileUtils.save_to_json("./city_bar_list.json", data)

    def extract_city_global_info(self, response):
        """
        获取 目的地信息
        :param response:
        :return:
        """
        el_title = response.xpath('//div[@class="title"]')
        mdd_title = el_title.xpath('//h1/text()').extract_first()
        mdd_photo_url = response.xpath('//div[@class="title"]/a/@href').extract_first()
        mdd_photo_number = el_title.xpath('//a/em/text()').extract_first()
        mdd_title_en = el_title.xpath('//span/text()').extract_first()

        info = {
            "mdd_title": mdd_title,
            "mdd_title_en": mdd_title_en,
            "mdd_photo_url": mdd_photo_url,
            "mdd_photo_number": mdd_photo_number,
        }

        return info

    def extract_city_sub_bar_info(self, response):
        """
        提取 子菜单数据
        :param response:
        :return:
        """
        el_bar_list = response.xpath("//div[contains(@class,'navbar-con')]/ul/li").extract()
        city_bar_list = []
        for index, info_bar in enumerate(el_bar_list):
            bar_sub_list = []
            el_hot_city = Selector(text=info_bar, type="html")
            bar_name = el_hot_city.xpath('.//a/span/text()').extract_first()
            bar_url = el_hot_city.xpath('.//a/@href').extract_first()
            bar_sub = el_hot_city.xpath('.//div').extract_first()
            if bar_sub is not None:
                bar_sub_text_list = el_hot_city.xpath(".//li/a/text()").extract()
                bar_sub_url_list = el_hot_city.xpath(".//li/a/@href").extract()

                for idx, bar_sub in enumerate(bar_sub_text_list):
                    bar_sub_list.append({"name": bar_sub, "href": bar_sub_url_list[idx]})

            info = {
                "bar_name": bar_name,
                "bar_url": bar_url,
                "bar_sub": bar_sub_list,
            }
            city_bar_list.append(info)
        return city_bar_list

    def extract_city_sales_list(self, response):
        """
        提取 旅行商城 数据
        :param response:
        :return:
        """
        el_bar_list = response.xpath("//ul[contains(@class,'sales-list')]/li").extract()
        city_bar_list = []
        for index, info_bar in enumerate(el_bar_list):
            el_hot_city = Selector(text=info_bar)
            bar_url = el_hot_city.xpath('.//a/@href').extract_first()
            img_url = el_hot_city.xpath('.//a/div[0]/img/@src').extract_first()
            img_text = el_hot_city.xpath('.//a/div[0]/div/text()').extract_first()
            bar_name = el_hot_city.xpath("//div[contains(@class,'caption')]/h3/text()").extract_first()
            bar_sell = el_hot_city.xpath("//span[contains(@class,'sell')]/text()").extract_first()
            bar_price = el_hot_city.xpath("//span[contains(@class,'price')]/text()").extract_first()

            info = {
                "bar_name": bar_name,
                "bar_url": bar_url,
                "img_url": img_url,
                "img_text": img_text,
                "bar_sell": bar_sell,
                "bar_price": bar_price,
            }
            city_bar_list.append(info)
        return city_bar_list

    def extract_city_overview_bar_info(self, response):
        """
        提取 概况菜单 数据
        :param response:
        :return:
        """
        el_bar_list = response.xpath("//div[contains(@class,'overview-drop')]/div/dl").extract()
        city_bar_list = []
        for index, info_bar in enumerate(el_bar_list):
            bar_sub_list = []
            el_hot_city = Selector(text=info_bar)
            bar_name = el_hot_city.xpath('.//dt/a/text()').extract_first()
            bar_url = el_hot_city.xpath('.//dt/a/@href').extract_first()

            bar_sub_text_list = el_hot_city.xpath('.//dd/a/text()').extract()
            bar_sub_href_list = el_hot_city.xpath('.//dd/a/@href').extract()
            for idx, (name, href) in enumerate(zip(bar_sub_text_list, bar_sub_href_list)):
                bar_sub_list.append({"name": name, "href": href})

            info = {
                "bar_name": bar_name,
                "bar_url": bar_url,
                "bar_sub": bar_sub_list,
            }
            city_bar_list.append(info)
        return city_bar_list


if __name__ == '__main__':
    pass

    url_list = {
        "mdd": "http://www.mafengwo.cn/mdd/",
        "china_city_info": "http://www.mafengwo.cn/mdd/citylist/21536.html",
        "city1": "http://www.mafengwo.cn/travel-scenic-spot/mafengwo/15693.html",
        "city2": "http://www.mafengwo.cn/travel-scenic-spot/mafengwo/10122.html",
        "city_yn": "http://www.mafengwo.cn/travel-scenic-spot/mafengwo/12711.html",
    }

    # url = url_list["mdd"]
    url = url_list["city_yn"]

    name = FileUtils.get_file_name(url)

    # get_url_html(url)

    mfw_spider = MafengwoSpider(url=url)
    # mfw_spider.get_city_list()
    mfw_spider.get_city_info()
