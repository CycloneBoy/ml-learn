#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : mafengwo_travel_note.py
# @Author: sl
# @Date  : 2022/4/9 - 下午3:36
from typing import List
from urllib.parse import urlparse

from attr import dataclass

from util.v2.file_utils import FileUtils


class AlbumPhoto:
    """
    游记图片详情
    """

    def __init__(self, travel_id=None, image_id=None, action="getAlbumPhoto", data=None):
        self.travel_id = travel_id
        self.image_id = image_id
        self.action = action

        self.vote_num = 0
        self.reply_num = 0
        self.share_num = 0
        self.is_vote = 0
        self.original_url = None
        self.poi = None
        self.replys = None

        if data is not None:
            self.parse_from_json(data)

    def is_success(self):
        return self.original_url is not None

    def get_image_name(self):
        parse_url = urlparse(self.original_url)
        file_name = parse_url.path
        file_name = FileUtils.get_file_name(file_name)

        return file_name

    def get_like_count(self, reply_weight=5, share_weight=3):
        """
        获取喜欢指数

        :param reply_weight:
        :param share_weight:
        :return:
        """
        count = self.vote_num + reply_weight * self.reply_num + share_weight * self.share_num
        return count

    def parse_from_json(self, raw_data):
        """
        解析
        :param raw_data:
        :return:
        """
        self.vote_num = raw_data["vote_num"]
        self.reply_num = raw_data["reply_num"]
        self.share_num = raw_data["share_num"]
        self.is_vote = raw_data["is_vote"]
        self.travel_id = raw_data.get("travel_id", None)
        self.image_id = raw_data.get("image_id", None)
        self.original_url = raw_data.get("original_url", None)
        self.poi = raw_data.get("poi", None)
        self.replys = raw_data.get("replys", None)

    def to_request(self):
        response = {
            "sAction": self.action,
            "iAlid": self.image_id,
            "iIid": self.travel_id,
        }
        return response

    def to_json(self):
        response = {
            "travel_id": self.travel_id,
            "image_id": self.image_id,
            "replys": self.replys,
            "vote_num": self.vote_num,
            "reply_num": self.reply_num,
            "share_num": self.share_num,
            "original_url": self.original_url,
            "poi": self.poi,
            "is_vote": self.is_vote,
        }
        return response

    def __str__(self):
        return str(self.to_json())


@dataclass
class TravelImage(object):
    """
    游记一张图片信息
    """
    src: str = None
    url: str = None
    album_photo: AlbumPhoto = None

    def parse_form_json(self, raw_data):
        """
        解析
        :param raw_data:
        :return:
        """
        self.src = raw_data["src"]
        self.url = raw_data["url"]
        self.album_photo = AlbumPhoto(data=raw_data["album_photo"])

    def get_like_count(self, reply_weight=5, share_weight=3):
        """
        获取喜欢指数

        :param reply_weight:
        :param share_weight:
        :return:
        """
        return self.album_photo.get_like_count(reply_weight=reply_weight, share_weight=share_weight)

    def get_up_count(self):
        count = [self.album_photo.vote_num, self.album_photo.share_num, self.album_photo.reply_num]
        return count

    def to_json(self):
        response = {
            "src": self.src,
            "url": self.url,
            "album_photo": self.album_photo.to_json(),
        }
        return response

    def __str__(self):
        return str(self.to_json())


@dataclass
class TravelNoteAllImageInfo(object):
    """
    游记所有图片信息
    """
    url: str = None
    travel_note_name: str = None
    image_count: str = None
    all_image_list: List[TravelImage] = []

    def parse_form_json(self, raw_data):
        """
        解析
        :param raw_data:
        :return:
        """
        self.url = raw_data["url"]
        self.travel_note_name = raw_data["travel_note_name"]
        self.image_count = raw_data["image_count"]

        all_image_list = raw_data["all_image_list"]

        all_image_list_item = []
        for item in all_image_list:
            travel_image = TravelImage()
            travel_image.parse_form_json(item)

            all_image_list_item.append(travel_image)

        self.all_image_list = all_image_list_item

    def to_json(self):
        response = {
            "url": self.url,
            "travel_note_name": self.travel_note_name,
            "image_count": self.image_count,
        }

        all_image_list_item = []
        for item in self.all_image_list:
            all_image_list_item.append(item.to_json())

        response["all_image_list"] = all_image_list_item

        return response

    def __str__(self):
        return str(self.to_json())


@dataclass
class TravelNoteInfo(object):
    """
    游记信息
    """

    id = None  # 游记链接ID
    travel_url = None  # 游记链接
    travel_name = None  # 游记名称
    travel_type = None  # 游记分类 ： 宝藏 、 星级
    travel_summary = None  # 游记摘要
    travel_destination = None  # 游记目的地
    travel_destination_country = None  # 游记目的地国家
    travel_image_url = None  # 游记封面图片链接
    author_id = None  # 游记作者ID
    author_url = None  # 游记作者首页链接
    author_name = None  # 游记作者名称
    author_city = None  # 游记作者名称
    author_image_url = None  # 游记作者图像链接
    author_lv = None  # 游记作者图像链接
    post_time = None  # 游记作者图像链接

    travel_view_count = None  # 游记浏览总数
    travel_comment_count = None  # 游记评论总数
    travel_up_count = None  # 游记顶的总数
    crawl_status = None  # 是否已经爬取
    crawl_time = None  # 爬取时间
    travel_father_id = None  # 游记父亲id
    travel_id = None

    travel_home_image_url = None  # 是否已经爬取
    travel_summary_all = None  # 爬取时间
    travel_share_count = None  # 游记父亲id
    travel_collect_count = None
    travel_time = None
    travel_day = None
    travel_people = None
    travel_cost = None

    travel_country_name = None
    travel_country_url = None
    travel_country_image_url = None
    travel_country_image_count = None

    travel_word_count = None
    travel_image_count = None
    travel_help_person_count = None
    travel_help_person_dest = None
    travel_first_image_list_url = None
    travel_image_url_list = None


@dataclass
class TravelNoteAndImageInfo(object):
    """
    游记信息
    """
    travel_note_info: TravelNoteInfo = None
    travel_note_all_image_info: TravelNoteAllImageInfo = None
