#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : eudic_api.py
# @Author: sl
# @Date  : 2021/11/20 - 下午4:31

"""
欧陆词典API接口封装

    ref: http://my.eudic.net/OpenAPI/doc_api_study
"""
import json

from util.file_utils import read_to_text
from util.logger_utils import logger
import requests

APP_TOKEN_EU = ""


class Category:
    """
    生词本
    """

    def __init__(self, name, language="en", cid=None):
        self.cid = cid
        self.name = name
        self.language = language

    def to_json(self):
        response = {"language": self.language, "name": self.name}
        if self.cid is not None:
            response["id"] = self.cid
        return response

    def __str__(self):
        return str(self.to_json())


class Words:
    """
    单词
    """

    def __init__(self, cid, words, language="en", name=None):
        self.cid = cid
        self.name = name
        self.language = language
        self.words = words

    def to_json(self):
        response = {"id": self.cid, "language": self.language, "words": self.words}
        return response


def get_token(file_name="/home/sl/workspace/python/a2020/ml-learn/test/pdf/eudic_token.txt"):
    token = read_to_text(file_name)
    logger.info(f"token: {token}")
    return token


def parse_categories(response):
    """
    parse category
    :param response:
    :return:
    """
    categories = []
    if response is None:
        return categories

    cats = json.loads(response)
    if 'data' in cats:
        cats = cats['data']
        if isinstance(cats, list):
            categories = [parse_category(cat) for cat in cats]
        else:
            categories = parse_category(cats)
    else:
        categories = parse_category(cats)
    return categories


def parse_category(data):
    return Category(cid=data['id'], name=data['name'], language=data['language'])


def parse_words(response, cid, language="en"):
    """
    parse category
    :param response:
    :return:
    """
    categories = []
    if response is None:
        return categories

    words = json.loads(response)
    if 'data' in words:
        words = Words(cid=cid, language=language, words=words['data'])
    return words


class EuApi:

    def __init__(self):
        self.api_util = EuDictApi(token=get_token())
        self.api_list = {
            "category": {
                "name": "获取所有生词本",
                "url": "https://api.frdic.com/api/open/v1/studylist/category",
                "method": "GET",
            },
            "words": {
                "name": "添加单词到生词本",
                "url": "https://api.frdic.com/api/open/v1/studylist/words",
                "method": "POST",
            }
        }

    def get_category(self, language="en"):
        url = "{}?language={}".format(self.api_list["category"]['url'], language)
        res = self.api_util.send_http_request(url=url, method="GET")
        categories = parse_categories(res)
        return categories

    def add_category(self, name="新增生词本1", language="en"):
        url = "{}".format(self.api_list["category"]['url'])
        data = Category(name=name, language=language).to_json()
        res = self.api_util.send_http_request(url=url, data=data, method="POST")
        categories = parse_categories(res)
        return categories

    def modify_category(self, cid, name="新增生词本1", language="en"):
        url = "{}".format(self.api_list["category"]['url'])
        data = Category(cid=cid, name=name, language=language).to_json()
        res = self.api_util.send_http_request(url=url, data=data, method="PATCH")
        category = parse_categories(res)
        return category

    def delete_category(self, cid, name="新增生词本1", language="en"):
        url = "{}".format(self.api_list["category"]['url'])
        data = Category(cid=cid, name=name, language=language).to_json()
        res = self.api_util.send_http_request(url=url, data=data, method="DELETE")
        return res

    def get_words(self, cid, language="en", page=0, page_size=100):
        url = "{}/{}?id={}&language={}&page={}&page_size={}". \
            format(self.api_list["words"]['url'],
                   cid, cid, language, page, page_size)
        res = self.api_util.send_http_request(url=url, method="GET")
        words = parse_words(res, cid, language)
        return words

    def add_words(self, cid, words, language="en"):
        url = "{}".format(self.api_list["words"]['url'])
        data = Words(cid=cid, words=words, language=language).to_json()
        res = self.api_util.send_http_request(url=url, data=data, method="POST")
        return res

    def delete_words(self, cid, words, language="en"):
        url = "{}".format(self.api_list["words"]['url'])
        data = Words(cid=cid, words=words, language=language).to_json()
        res = self.api_util.send_http_request(url=url, data=data, method="DELETE")
        return res


class EuDictApi:
    def __init__(self, token):
        self.token = token
        self.header = self.build_header()

    def build_header(self):
        headers = {
            "Authorization": self.token,
            "Content-Type": "application/json",
        }
        return headers

    def send_http_request(self, url, data=None, method='GET'):
        """发送http 请求"""
        logger.info("开始发送请求：{} - {} : {}".format(method, url, data))
        if method.upper() == 'GET':
            if data is not None:
                r = requests.get(url, params=data, headers=self.header).content.decode("utf-8")
            else:
                r = requests.get(url, headers=self.header).content.decode("utf-8")
        elif method.upper() == 'POST':
            if data is not None:
                r = requests.post(url, json=data, headers=self.header).content.decode("utf-8")
            else:
                r = requests.post(url, headers=self.header).content.decode("utf-8")
        elif method.upper() == 'PATCH':
            if data is not None:
                r = requests.patch(url, json=data, headers=self.header).content.decode("utf-8")
            else:
                r = requests.patch(url, headers=self.header).content.decode("utf-8")
        elif method.upper() == 'DELETE':
            if data is not None:
                r = requests.delete(url, json=data, headers=self.header).content.decode("utf-8")
            else:
                r = requests.delete(url, headers=self.header).content.decode("utf-8")
        else:
            r = ""

        logger.info("请求的返回结果：{}".format(r))
        return r


if __name__ == '__main__':
    pass
    # get_token()

    euapi = EuApi()
    words = ["modify", "category"]
    res = euapi.get_category()
    # res = euapi.add_category(name="测试2")
    # res = euapi.modify_category(uid="132818869442535704", name="测试3")
    # res = euapi.delete_category(uid="132818869442535704", name="测试3")
    # res = euapi.delete_category(uid="132818750585486525", name="测试")
    # print(res)
    # res = euapi.get_words(uid="132818750585486525")
    # res = euapi.add_words(uid="132818750585486525", words=words)
    # res = euapi.delete_words(uid="132818750585486525", words=words)
    print(res)
    res = euapi.get_category()
    print(res)
