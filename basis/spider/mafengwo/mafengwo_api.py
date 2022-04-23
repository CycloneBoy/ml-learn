#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : mafengwo_api.py
# @Author: sl
# @Date  : 2022/4/9 - 上午9:45
import json
import os
import random
import time
import traceback
from copy import deepcopy

from basis.spider.mafengwo.entity.mafengwo_travel_note import AlbumPhoto, TravelNoteAllImageInfo
from basis.utils.random_user_agent import RandomUserAgentMiddleware
from util.logger_utils import logger
from util.v2.constants import Constants
from util.v2.file_utils import FileUtils
from util.v2.http_utils import HttpUtils

"""
mafengwo json api

"""


class MafengwoApi(object):
    """
    请求 API
    """

    def __init__(self, url, download_dir=Constants.SPIDER_MAFENGWO_DIR, host_name="mafengwo"):
        self.url = url
        self.download_dir = download_dir
        self.host_name = host_name
        self.user_cookies = self.read_cookie()
        self.use_simple_header = False
        # self.use_simple_header = False
        self.header = self.build_header()

    def read_cookie(self, file_name=Constants.MAFENGWO_COOKIE_FILE):
        line = FileUtils.read_to_text_list(file_name)[1]
        self.user_cookies = line
        return line

    def build_header(self, url=None):
        headers = {
            "host": "www.mafengwo.cn",
            "cache-control": "max-age=0",
            # "user-agent": RandomUserAgentMiddleware().get_user_agent(),
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "cookie": self.user_cookies,
            "connection": "keep-alive",
            "referer": "http://www.mafengwo.cn/travel-scenic-spot/mafengwo/10460.html",
            "Accept-Encoding": "gzip, deflate",
            "Pragma": "no-cache",
            "X-Requested-With": "XMLHttpRequest",
        }

        if self.use_simple_header:
            headers = {
                "host": "www.mafengwo.cn",
                "accept-encoding": "gzip, deflate, br",
                "cache-control": "no-cache",
                # "user-agent": RandomUserAgentMiddleware().get_user_agent(),
                "user-agent": "PostmanRuntime/7.26.8",
                "accept": "*/*",
                "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
                # "cookie": '__jsluid_h=72ccbcaa1094196f0f61f5afd9dc03d6; mfw_uuid=62511a6b-fac0-3f5d-7c5f-51c6b394e85a; oad_n=a%3A3%3A%7Bs%3A3%3A%22oid%22%3Bi%3A1029%3Bs%3A2%3A%22dm%22%3Bs%3A15%3A%22www.mafengwo.cn%22%3Bs%3A2%3A%22ft%22%3Bs%3A19%3A%222022-04-09+13%3A32%3A27%22%3B%7D; PHPSESSID=g9trkla350j7cd5hjlpoalct23',
                "connection": "keep-alive",
                "Pragma": "no-cache",
            }
        return headers

    def sent_request(self, url, data=None, method='GET', encode="utf-8", **kwargs):
        """
        获取 页面

        :param url:
        :param data:
        :param method:
        :param encode:
        :return:
        """
        header = self.build_header(url)
        # logger.info(f"header:{header}")
        html = HttpUtils.send_http_request(url=url, header=header, data=data, method=method, encode=encode, **kwargs)
        if not str(html).startswith("<!DOCTYPE html>") and str(html).startswith("{"):
            html = json.loads(html)

        return html


class MafengwoBusinessApi(object):

    def __init__(self):
        self.debug = True
        self.api_util = MafengwoApi(url=None)
        self.api_list = {
            "getAlbumPhoto": {
                "name": "获取游记图片详情",
                "url": Constants.SPIDER_MFW_API_GET_AJAX_ANY,
                "method": "GET",
            }
        }

    def get_album_photo_info(self, url, retry=False):
        """
        获取游记图片详情
        http://www.mafengwo.cn/photo/18671/scenery_23513200/1694284233.html

        http://www.mafengwo.cn/mdd/photo/ajax_any.php?sAction=getAlbumPhoto&iIid=23513200&iAlid=1682347451

        :param url:
        :param retry:
        :return:
        """
        image_id = FileUtils.get_url_file_name(url)
        travel_id = FileUtils.get_url_file_parent_name(url)
        travel_id = travel_id.split("_")[1]

        album_photo = AlbumPhoto(travel_id=travel_id, image_id=image_id)

        save_file_name = FileUtils.get_album_photo_path(url=url)
        if not retry and self.check_json_exist(save_file_name):
            res = FileUtils.load_to_json(save_file_name)
            logger.info(f"获取游记图片详情,从文件中读取: success:{url} -> {save_file_name}")
        else:
            # 重新获取
            data = album_photo.to_request()
            try:
                res = self.api_util.sent_request(url=Constants.SPIDER_MFW_API_GET_AJAX_ANY, data=data,
                                                 method="GET", timeout=3)
            except Exception as e:
                traceback.print_exc()
                logger.error(f"获取游记图片详情失败:{data}")
                res = {}
            if isinstance(res, dict):
                FileUtils.save_to_json(save_file_name, res)

        # 解析
        success = False
        if "payload" in res:
            info = res["payload"]["photo"][str(image_id)]
        else:
            return album_photo

        if "original_url" in info:
            success = True

        logger.info(f"获取游记图片详情success:{success}:{url} -> {save_file_name}")

        album_photo.parse_from_json(info)

        return album_photo

    def check_json_exist(self, file_path):
        flag = False
        if FileUtils.check_file_exists(file_path):
            content = FileUtils.load_to_json(file_path)
            if "payload" not in content:
                flag = False
            else:
                flag = True

        return flag

    def get_travel_note_info(self, url, max_sleep=5, min_count=1):
        """
            获取游记信息

             - 获取游记详情页面信息
             - 获取游记所有图片信息

        :param url:
        :param max_sleep:
        :param min_count:
        :return:
        """
        current_file_name = FileUtils.get_url_file_name(url)
        file_name_travel_note_info = f"{Constants.SPIDER_MFW_TRAVEL_INFO_DIR}/{current_file_name}.json"

        # 获取游记详情页面信息
        travel_note_info = FileUtils.load_to_json(file_name_travel_note_info)

        # 获取游记图片信息
        travel_first_image_list_url = travel_note_info["travel_first_image_list_url"]

        file_name_scenery_image_url = travel_first_image_list_url[
                                      : str(travel_first_image_list_url).rindex("/")] + "_1.html"

        file_name_scenery_name = FileUtils.get_url_file_name(file_name_scenery_image_url)
        file_name_travel_image_info = f"{Constants.SPIDER_MFW_TRAVEL_IMAGE_INFO_DIR}/{file_name_scenery_name}.json"

        # 获取游记所有图片信息
        travel_image_info = FileUtils.load_to_json(file_name_travel_image_info)

        # 获取游记所有图片 详情 信息
        self.get_travel_note_all_album_photo_info(file_name_travel_image_info, max_sleep=max_sleep)

        # 获取游记所有图片 详情 信息 topk
        travel_note_all_image_info_topk = self.get_all_travel_image_info_topk(travel_image_info=travel_image_info,
                                                                              min_count=min_count)

        return travel_note_info

    def get_travel_note_all_album_photo_info(self, file_name_travel_image_info, max_sleep=5):
        """
        获取游记所有图片 详情 信息

        :param file_name_travel_image_info:
        :param max_sleep:
        :return:
        """
        travel_image_info = FileUtils.load_to_json(file_name_travel_image_info)

        url = travel_image_info["url"]
        travel_note_name = travel_image_info["travel_note_name"]
        travel_image_list = travel_image_info["all_image_list"]

        logger.info(f"获取游记所有图片详情信息:url:{url} - {travel_note_name} - {len(travel_image_list)}")

        error_info = []
        for index, item in enumerate(travel_image_list):
            image_src = item["src"]

            if self.debug:
                logger.info(f"获取:{index} - url: {image_src}")

            if "album_photo" not in item:
                sleep_time = random.random() * max_sleep
                logger.info(f"sleep: {sleep_time}")
                time.sleep(sleep_time)

                album_photo = self.get_album_photo_info(url=image_src)
                if album_photo.is_success():
                    item["album_photo"] = album_photo.to_json()
                    FileUtils.save_to_json(file_name_travel_image_info, travel_image_info)
                else:
                    error_info.append([index, image_src])
                    logger.info(f"获取失败 {index} - url: {image_src}")

        logger.info(f"获取游记所有图片详情信息完毕,获取失败列表: {len(error_info)}")
        for item in error_info:
            logger.info(f"{item}")

        return error_info

    def get_all_travel_image_info_topk(self, travel_image_info, topk=-1, min_count=1):
        """
        对游记所有图片详情信息 进行排序
            - 根据点赞,收擦和回复数量
        :param travel_image_info:
        :param topk:
        :param min_count:
        :return:
        """
        travel_note_all_image_info = TravelNoteAllImageInfo()
        travel_note_all_image_info.parse_form_json(travel_image_info)

        file_name_scenery_name = FileUtils.get_url_file_name(travel_note_all_image_info.url)
        travel_note_id = file_name_scenery_name.split("_")[1]

        # print(travel_note_all_image_info)

        sort_result = []
        for index, travel_image in enumerate(travel_note_all_image_info.all_image_list):
            album_photo = travel_image.album_photo

            if album_photo.get_like_count() >= min_count:
                sort_result.append(travel_image)

        sort_result.sort(key=lambda x: x.get_like_count(), reverse=True)

        # 拷贝到指定位置
        topk_dir = f"{Constants.SPIDER_MFW_TRAVEL_IMAGE_TOPK_DIR}/{travel_note_id}/{file_name_scenery_name}_{min_count}"

        for index, travel_image in enumerate(sort_result):
            up_count = travel_image.get_up_count()
            up_count_prefix = "_".join([str(i) for i in up_count])
            logger.info(f"{index} - {travel_image.src} - {up_count}")

            image_name = FileUtils.get_file_name(travel_image.url)
            src_image_file_name = f"{Constants.SPIDER_MFW_TRAVEL_IMAGE_DIR}/{travel_note_id}/{image_name}"

            new_name = f"{topk_dir}/{up_count_prefix}_{image_name}"
            FileUtils.copy_file(src_image_file_name, dst=new_name)

        # 保存 topk
        travel_note_all_image_info_topk = deepcopy(travel_note_all_image_info)
        travel_note_all_image_info_topk.all_image_list = sort_result

        data = travel_note_all_image_info_topk.to_json()

        file_name_travel_image_topk_info = f"{Constants.SPIDER_MFW_TRAVEL_IMAGE_INFO_TOPK_DIR}/{file_name_scenery_name}_{min_count}.json"
        FileUtils.save_to_json(file_name_travel_image_topk_info, data)
        logger.info(f"保存游记所有图片详情信息topk: total:{len(sort_result)} - {file_name_travel_image_topk_info} ")

        return travel_note_all_image_info_topk


def demo_spider():
    # url = "http://www.mafengwo.cn/travel-scenic-spot/mafengwo/10460.html"
    url = "http://www.mafengwo.cn/mdd/photo/ajax_any.php?sAction=getAlbumPhoto&iAlid=1694284234&iIid=23513200"
    spider_base = MafengwoApi(url=url)
    html = spider_base.sent_request(url)
    if isinstance(html, dict):
        FileUtils.save_to_json("./test.json", html)
    FileUtils.save_to_text("./test.html", html)


def demo_get_image():
    url = "http://www.mafengwo.cn/photo/18671/scenery_23513200/1682347451.html"
    spider_base = MafengwoBusinessApi()
    # data = spider_base.get_album_photo_info(url)

    # url = "http://www.mafengwo.cn/i/23498023.html"
    # url = "http://www.mafengwo.cn/i/23513200.html"
    # url = "http://www.mafengwo.cn/i/21448504.html"
    # url = "http://www.mafengwo.cn/i/23497631.html"
    url = "http://www.mafengwo.cn/i/18292819.html"
    data = spider_base.get_travel_note_info(url, max_sleep=5, min_count=2)
    print(data)


if __name__ == '__main__':
    pass
    # demo_spider()
    demo_get_image()
