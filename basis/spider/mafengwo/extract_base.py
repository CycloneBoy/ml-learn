#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : extract_base.py
# @Author: sl
# @Date  : 2022/2/19 - 下午10:05
from abc import ABC
import os
import time
import traceback
from collections import OrderedDict
from typing import List

import requests
from docx import Document
from docx.shared import Inches

from fpdf import FPDF
from scrapy import Selector
from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.wait import WebDriverWait

import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from basis.utils.random_user_agent import RandomUserAgentMiddleware
from test.pdf.extract_pdf_word import extract_words_from_txt_and_to_dict
from util.file_utils import save_to_json, load_to_json, get_file_name, list_file, read_to_text_list
from util.logger_utils import logger
from util.v2.constants import Constants
from util.v2.file_utils import FileUtils

PATTERN_YOUTUBE_DOWNSUB = "https://downsub.com/?url=https%3A%2F%2Fyoutu.be%2F{}"
DOWNLOAD_DIR = "/home/sl/workspace/data/video/youtube/audio"
GLOBAL_VIDEO_LIST_DICT = f"{DOWNLOAD_DIR}/video_list_dict.json"


class ExtractSpiderBase(ABC):

    def __init__(self, url, download_dir=Constants.SPIDER_DIR, host_name="mafengwo"):
        self.url = url
        self.download_dir = download_dir
        self.host_name = host_name
        self.options = Options()
        self.driver = None
        self.file_name = None
        self.content = None
        self.response: Selector = None
        self.username = None
        self.password = None
        self.user_cookies = self.read_cookie()

    def get_driver(self, download_dir=None):
        if download_dir is None:
            download_dir = self.download_dir
        self.options.add_experimental_option('prefs', {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True
        })
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.implicitly_wait(300)
        # self.driver.maximize_window()
        self.driver.set_page_load_timeout(300)
        # self.driver.set_window_size(1124, 850)
        return self.driver

    def get_url_html(self, url=None, ):
        if self.driver is None:
            self.get_driver()
        if url is None:
            url = self.url
        self.driver.get(url)

        content = self.driver.page_source
        # time.sleep(5)
        self.driver.quit()

        self.file_name = self.get_html_file_path()
        FileUtils.save_to_text(self.file_name, content)
        return content

    def get_url_content(self, url=None, use_driver=False, retry=False):
        """
        获取url链接的html数据
        :param url:
        :param use_driver:
        :param retry:
        :return:
        """
        file_path = FileUtils.get_url_file_path(url)
        # file_path = self.get_html_file_path()
        if not retry and FileUtils.check_file_exists(file_path):
            content = FileUtils.get_content(file_path, encoding="utf-8")
        else:
            if use_driver:
                content = self.get_url_html(url=url, )
            else:
                content = self.get_url_html_by_request(url=url)
            FileUtils.save_to_text(file_path, content)

        self.content = content
        self.response = Selector(text=content)
        return content

    def get_html_file_path(self):
        file_name = FileUtils.get_file_name(self.url)
        return f'{self.download_dir}/{self.host_name}/{file_name}'

    def xpath(self, query):
        return self.response.xpath(query)

    def text(self, query):
        el_link = self.xpath(query)
        res = el_link.xpath("//text()").extract()
        return res

    def href(self, query):
        el_link = self.xpath(query)
        res = el_link.xpath("@href").extract()
        return res

    def css(self, query):
        return self.response.css(query)

    def re(self, query):
        return self.response.re(query)

    def read_username(self, use_token_file=Constants.MAFENGWO_TOKEN_FILE):
        line = FileUtils.read_to_text_list(use_token_file)[0]
        splits = line.split()
        self.username = splits[0]
        self.password = splits[1]

    def read_cookie(self, file_name=Constants.MAFENGWO_COOKIE_FILE):
        line = FileUtils.read_to_text_list(file_name)[1]
        self.user_cookies = line

    def get_url_html_by_request(self, url):
        header = self.build_header(url)
        html = requests.get(url, headers=header).content.decode("utf-8")

        return html

    def build_header(self, url=None):
        headers = {
            "host": "www.mafengwo.cn",
            "accept-encoding": "http://www.mafengwo.cn/",
            "cache-control": "max-age=0",
            "user-agent": RandomUserAgentMiddleware().get_user_agent(),
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "cookie": self.user_cookies,
            "connection": "keep-alive",
            "referer": "http://www.mafengwo.cn/travel-scenic-spot/mafengwo/10460.html",
            "Accept-Encoding": "gzip, deflate",
            "Pragma": "no-cachem{\
                      ",
        }

        return headers

    def filter_space(self, content):
        res = str(content).replace("\n", "").replace(" ", "")
        return res


def demo_spider():
    url = "http://www.mafengwo.cn/travel-scenic-spot/mafengwo/10460.html"
    spider_base = ExtractSpiderBase(url=url)
    html = spider_base.get_url_html_by_request(url)
    FileUtils.save_to_text("./test.html", html)


if __name__ == '__main__':
    pass
    demo_spider()
