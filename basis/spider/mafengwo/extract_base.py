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
        return self.driver

    def get_url_html(self, url=None,):
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

    def get_url_content(self, url=None,):
        """
        获取url链接的html数据
        :param url:
        :return:
        """
        file_path = self.get_html_file_path()
        if FileUtils.check_file_exists(file_path):
            content = FileUtils.get_content(file_path, encoding="utf-8")
        else:
            content = self.get_url_html(url=url,)

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
