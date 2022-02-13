#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : craw_list.py
# @Author: sl
# @Date  : 2020/11/11 - 下午10:00
import random
import os
import sys
import codecs
import shutil
import urllib
import re

from scrapy import Selector
from scrapy.http import HtmlResponse
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import  Keys
import selenium.webdriver.support.ui as ui
from selenium.webdriver.common.action_chains import ActionChains
import re
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ChromeOptions
import requests
from basis.utils.random_user_agent import RandomUserAgentMiddleware

import urllib.request
# import urllib3.request
from basis.utils.multi_thread_download import multi_download,stop_treads,exitFlag
# from database.mysql_db import AsyncMysqlPipeline
# from entity.items import MafengwoTravelDetail



MAFENGWO_HOST = "http://www.mafengwo.cn"
nameindex = 1

def getTravelNote(url):
    # 如果driver没加入环境变量中，那么就需要明确指定其路径
    # 验证于2017年4月11日
    # 直接登陆新浪微博
    chrome_options = ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    # browser = webdriver.Chrome(options=chrome_options)
    browser = webdriver.Chrome()
    browser.maximize_window()
    browser.set_page_load_timeout(30000)
    browser.set_window_size(1124, 850)

    try:
        browser.get(url)
        browser.execute_script('window.scrollTo(0, document.body.scrollHeight)')
        print(browser.title)
        print(browser.page_source)
        time.sleep(5)
        saveToFile("data.html",browser.page_source)
    except TimeoutException as e:
        print('超时')
        browser.execute_script('window.stop()')
    return browser.page_source



def saveToFile(filename,body):
    # if not os.path.exists(filename):
    #     os.mkdir(filename)
    with open(file=filename,mode='w',encoding='utf-8') as f:
        f.write(body)

def atest_chrome():
    chrome_driver = '/usr/local/bin/chromedriver'  # chromedriver的文件位置
    b = webdriver.Chrome(executable_path=chrome_driver)
    b.get('https://www.google.com')
    time.sleep(5)
    b.quit()

def atest_chrome2(url):
    chrome_driver = '/usr/local/bin/chromedriver'  # chromedriver的文件位置
    b = webdriver.Chrome()
    wait = WebDriverWait(b, 60)
    b.get(url)
    time.sleep(20)
    b.quit()

def atest_foxfire(url):
    chrome_driver = '/usr/local/bin/chromedriver'  # chromedriver的文件位置
    b = webdriver.Firefox()
    b.get(url)
    time.sleep(30)
    b.quit()

if __name__ == '__main__':
    # url = "http://www.mafengwo.cn/u/42370376/note.html"
    # url = "https://downsub.com/"
    url = "https://downsub.com/?url=https%3A%2F%2Fyoutu.be%2FMdUkC7Vz3rg"
    # url = "http://www.mafengwo.cn"
    # url = "http://www.baidu.com"
    # url = 'http://hotel.qunar.com/city/beijing_city/'
    # url = "https://www.google.com/"
    # url = "https://www.cnblogs.com/ytkahm"
    # html = getTravelNote(url=url)
    atest_chrome2(url)
    # test_foxfire(url)

    # url_image = 'background-image: url("https://statics0.umustatic.cn//videoweike/teacher/weike/jK1W627e/thumbnail/1581917553.4833.16784.6183267.mp4_00804.jpg");'
    #
    # print(url_image.index('url('))
    # print(url_image.index('.mp4'))
    # parse_url = url_image[url_image.index('http'):url_image.index('.mp4')+ 4]
    # # parse_url = re.findall(r'^(http|https)://.*',url_image)
    # print(parse_url)