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

from basis.spider.mafengwo.entity.mafengwo_items import MafengwoTravel
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

    def get_city_info(self, url=None, use_driver=False, retry=False):
        """
        get city list

        :param url:  http://www.mafengwo.cn/travel-scenic-spot/mafengwo/12711.html
        :return:
        """
        # file_name = FileUtils.get_url_file_path(url)
        # content = FileUtils.get_content(file_name, encoding="utf-8")
        content = self.get_url_content(url, use_driver=use_driver, retry=retry)

        response = Selector(text=content)

        mdd = response.xpath('//*[@id="container"]/div[1]/div/div[2]/div[2]/div/span/a').extract_first()

        city_global_info = self.extract_city_global_info(response)
        city_bar_list = self.extract_city_sub_bar_info(response)
        city_sales_list = self.extract_city_sales_list(response)
        city_overview_bar_list = self.extract_city_overview_bar_info(response)
        city_gonglve_list = self.extract_city_gonglve_pdf_list(response)
        city_note_list = self.extract_city_note_list(response)

        data = {
            "city_global_info": city_global_info,
            "city_bar_list": city_bar_list,
            "city_sales_list": city_sales_list,
            "city_overview_bar_list": city_overview_bar_list,
            "city_gonglve_list": city_gonglve_list,
            "city_note_list": city_note_list,
        }

        logger.info(f'{city_bar_list}')
        current_file_name = FileUtils.get_url_file_name(url)
        city_bar_file_name = f"{Constants.SPIDER_MFW_CITY_INFO_DIR}/{current_file_name}.json"
        FileUtils.save_to_json(city_bar_file_name, data)
        FileUtils.copy_file(city_bar_file_name, Constants.DIR_DATA_JSON_CITY_INFO)

        success = True if len(city_bar_list) > 0 else False
        return data, success

    def extract_city_global_info(self, response):
        """
        获取 目的地信息
        :param response:
        :return:
        """
        el_title = response.xpath('//div[@class="title"]')
        mdd_title = el_title.xpath('.//h1/text()').extract_first()
        mdd_photo_url = response.xpath('.//div[@class="title"]/a/@href').extract_first()
        mdd_photo_number = el_title.xpath('.//a/em/text()').extract_first()
        mdd_title_en = el_title.xpath('.//span/text()').extract_first()

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
                    try:
                        bar_sub_list.append({"name": bar_sub, "href": bar_sub_url_list[idx]})
                    except Exception as e:
                        traceback.print_exc()
                        print(bar_sub_text_list)
                        print(bar_sub_url_list)

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
            img_url = el_hot_city.xpath(".//div[contains(@class,'image')]/img/@src").extract_first()
            img_text = el_hot_city.xpath(".//div[contains(@class,'image')]/div/text()").extract_first()
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

    def extract_city_gonglve_pdf_list(self, response):
        """
        提取 攻略下载 数据
        :param response:
        :return:
        """
        el_bar_list = response.xpath("//ul[contains(@class,'gl-list')]/li").extract()
        city_bar_list = []
        for index, info_bar in enumerate(el_bar_list):
            el_hot_city = Selector(text=info_bar)
            bar_url = el_hot_city.xpath(".//a[contains(@class,'gl-cover')]/@href").extract_first()
            img_url = el_hot_city.xpath(".//a[contains(@class,'gl-cover')]/img/@src").extract_first()
            pdf_size = el_hot_city.xpath("//span[1]/text()").extract_first()
            jpeg_size = el_hot_city.xpath("//span[2]/text()").extract_first()
            info = {
                "bar_url": bar_url,
                "img_url": img_url,
                "pdf_size": str(pdf_size).replace("文件大小", ""),
                "jpeg_size": str(jpeg_size).replace("文件大小", ""),
            }
            city_bar_list.append(info)
        return city_bar_list

    def extract_city_note_list(self, response):
        """
        提取 游记 数据
        :param response:
        :return:
        """
        city_bar_list = self.parse_note(response)

        return city_bar_list

    def parse_note(self, response, hot_travel_destination=None):
        """
        提取游记信息
        :param response:
        :param hot_travel_destination:
        :return:
        """
        # response = Selector(text=text)

        # 获取游记列表
        travel_note_list = response.xpath('//div[contains(@class,"tn-item clearfix")]')

        travel_note_list_index = 0
        travel_list = []
        for travel_note in travel_note_list:
            travel_note_list_index += 1
            # print("游记列表：" + str(travel_note_list_index) + "  "+str(travel_note))

            try:
                travel_image_url = travel_note.xpath('div[1]/a/img/@data-original').extract_first()
                travel_type = travel_note.xpath('div[@class="tn-image"]/a/@title').extract_first()
                travel_name = travel_note.xpath('div[2]/dl/dt/a[@class="title-link"]/text()').extract_first()
                travel_url = travel_note.xpath('div[2]/dl/dt/a[@class="title-link"]/@href').extract_first()
                travel_summary = str(travel_note.xpath('div[2]/dl/dd/a/text()').extract_first()).replace('\n', "") \
                    .replace(' ', "")
                travel_up_count = travel_note.xpath('div[2]/div/span/a/@data-vote').extract_first()
                author_url = travel_note.xpath('div[2]/div/span[2]/a/@href').extract_first()
                author_name = travel_note.xpath('div[2]/div/span[2]/a/text()').extract_first()

                author_image_url = travel_note.xpath('div[2]/div/span[2]/a/img/@src').extract_first().split("?")[0]
                travel_view_count = travel_note.xpath('div[2]/div/span[3]/text()').extract_first().split("/")[0]
                travel_comment_count = travel_note.xpath('div[2]/div/span[3]/text()').extract_first().split("/")[1]

                # travel = MafengwoTravel()
                travel = {}
                travel["travel_image_url"] = travel_image_url
                travel["travel_type"] = travel_type
                travel["travel_name"] = travel_name
                travel["travel_url"] = travel_url
                travel["travel_summary"] = travel_summary
                travel["travel_up_count"] = int(travel_up_count)
                travel["author_name"] = author_name
                travel["author_image_url"] = author_image_url
                travel["travel_view_count"] = int(travel_view_count)
                travel["travel_comment_count"] = int(travel_comment_count)
                travel["crawl_status"] = "0"
                # travel["travel_destination"] = hot_travel_destination['city_name']
                # travel["travel_destination_country"] = hot_travel_destination['country_name']
                # travel["travel_father_id"] = str(hot_travel_destination['id'])
                travel["travel_id"] = travel_url.split("/")[2][: -5]
                travel["author_id"] = author_url.split("/")[2][: -5]

                if travel_url is not None:
                    travel["travel_url"] = 'http://www.mafengwo.cn' + travel_url

                if author_url is not None:
                    travel["author_url"] = 'http://www.mafengwo.cn' + author_url

                if travel_type is None:
                    travel["travel_type"] = "user"

                # print()
                # print("游记列表：" + str(travel_note_list_index) + " -> " + str(travel))
                logger.debug("游记列表：" + str(travel_note_list_index) + " -> " + str(travel))
                # print()
                travel_list.append(travel)
            except BaseException as e:
                logger.error("parse_note 解析出错！原因：%s" % e.args)
                print("parse_note 解析出错！原因：%s" % e.args)
                continue
        return travel_list

    @staticmethod
    def get_url_by_city_name(hot_city_list, city_name):
        url = ""
        url_index = -1
        for index, data in enumerate(hot_city_list):
            if str(data["name"]).startswith(city_name):
                url = data["href"]
                url_index = index
                break
        return url, url_index


if __name__ == '__main__':
    pass
    hot_city_list_file_name = "./hot_city_list.json"
    hot_city_list = FileUtils.load_to_json(hot_city_list_file_name)

    url_list = {
        "mdd": "http://www.mafengwo.cn/mdd/",
        "china_city_info": "http://www.mafengwo.cn/mdd/citylist/21536.html",
        "city_ukn": "http://www.mafengwo.cn/travel-scenic-spot/mafengwo/15693.html",
        "city_yg": "http://www.mafengwo.cn/travel-scenic-spot/mafengwo/10122.html",
        "city_yn": "http://www.mafengwo.cn/travel-scenic-spot/mafengwo/12711.html",
    }

    # url = url_list["mdd"]
    url = url_list["city_yn"]
    # url = url_list["city_ukn"]
    # url = url_list["city_yg"]

    name = FileUtils.get_file_name(url)

    # get_url_html(url)

    mfw_spider = MafengwoSpider(url=url)
    # mfw_spider.get_city_list()
    # mfw_spider.get_city_info(url)
    # mfw_spider.get_url_content(url)

    # res = FileUtils.get_url_file_path(url="/photo/mdd/12711.html")
    # res = FileUtils.get_url_file_name(url="/photo/mdd/12711.html")
    # print(res)

    city_list = ["city_yn", "city_ukn", "city_yg"]
    # for key in city_list:
    #     mfw_spider.get_city_info(url=url_list[key])

    need_url, url_index = MafengwoSpider.get_url_by_city_name(hot_city_list=hot_city_list, city_name="东京")
    mfw_spider.get_city_info(url=need_url)

    for index, data in enumerate(hot_city_list):
        try:
            if "status" in data and data['status'] == 1:
                logger.info(f"down : {index} - {data['name']} - {url}")
                continue
            name = data["name"]
            url = f"http://www.mafengwo.cn" + data["href"]
            logger.info(f"begin : {index} - {name} - {url}")
            data, success = mfw_spider.get_city_info(url=url)

            if not success:
                data, success = mfw_spider.get_city_info(url=url, retry=True, use_driver=True)
            if success:
                data["status"] = 1
            else:
                data["status"] = 0
            FileUtils.save_to_json(hot_city_list_file_name, hot_city_list, )
        except Exception as e:
            traceback.print_exc()
            print(f"error:{index} - {data}")
