#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : http_utils.py
# @Author: sl
# @Date  : 2022/4/9 - 上午10:09
import requests

from util.logger_utils import logger
from util.v2.common_utils import BaseUtils


class HttpUtils(BaseUtils):

    @staticmethod
    def send_http_request(url, header, data=None, method='GET', encode="utf-8"):
        """
        发送http 请求
        
        :param url: 
        :param header: 
        :param data: 
        :param method: 
        :param encode:
        :return:
        """

        proxy = '118.163.120.181:58837'
        proxies = {
            'http': 'http://' + proxy,
            'https': 'https://' + proxy,
        }
        proxies = None
        logger.info("开始发送请求：{} - {} : {}".format(method, url, data))
        if method.upper() == 'GET':
            # r = requests.get(url, params=data, headers=header).content.decode(encode)
            contents = requests.get(url, params=data, headers=header)
            r = contents.content.decode(encode)
        elif method.upper() == 'POST':
            r = requests.post(url, json=data, headers=header).content.decode(encode)
        elif method.upper() == 'PATCH':
            r = requests.patch(url, json=data, headers=header).content.decode(encode)
        elif method.upper() == 'DELETE':
            r = requests.delete(url, json=data, headers=header).content.decode(encode)
        else:
            r = ""

        logger.info("请求的返回结果：{}".format(r))

        return r
