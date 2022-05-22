#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project ml-learn
# @File  : parse_home.py
# @Author: sl
# @Date  : 2022/5/20 - 下午8:17
import json
import random
import re
import time
import traceback
import urllib
import urllib.request
from copy import deepcopy

from urllib import parse
from Crypto.Cipher import AES
import execjs

import requests
from scrapy import Selector

from basis.spider.download_v1 import download_v2
from basis.spider.mafengwo.extract_base import ExtractSpiderBase
from basis.utils.multi_thread_download_v2 import MultiTheadDownloader
from basis.utils.random_user_agent import RandomUserAgentMiddleware
from util.logger_utils import logger
from util.v2.constants import Constants
from util.v2.file_utils import FileUtils
from util.v2.http_utils import HttpUtils


class CourseSpider(ExtractSpiderBase):

    def __init__(self, url=None, use_driver=False, retry=False):
        super().__init__(url)
        self.use_driver = use_driver
        self.retry = retry
        self.config = self.load_config(file_name=Constants.COURSE_CONFIG_FILE)
        self.config_env = self.config['env']
        self.headers = self.config_env['header']
        self.user_cookies = self.read_cookie(file_name=self.config_env['cookie_path'])

        self.config_spider = self.config['spider']
        self.app_id = self.config_spider['app_id']
        self.product_id = self.config_spider['product_id']
        self.resource_id = self.config_spider['resource_id']

        self.base_url = self.config_spider['base_url']
        self.course_list_url = self.base_url + self.config_spider['course_list_url']
        self.video_info_url = self.base_url + self.config_spider['video_info_url']
        self.column_info_url = self.base_url + self.config_spider['column_info_url']
        self.course_list_v2_url = self.base_url + self.config_spider['course_list_v2_url']
        self.course_web_url = self.base_url + self.config_spider['course_web_url']

        self.total_page = self.config_spider['total_page']
        self.video_url_prefix = self.config_spider['video_url_prefix']

        self.config_run_args = self.config['run_args']
        self.out_dir = self.config_run_args["out_dir"]
        self.out_json_dir = self.config_run_args["out_json_dir"]
        self.out_audio_dir = self.config_run_args["out_audio_dir"]
        self.out_video_dir = self.config_run_args["out_video_dir"]
        self.out_video_dir_big = self.config_run_args["out_video_dir_big"]
        self.out_web_video_dir = self.config_run_args["out_web_video_dir"]

    def get_home_list(self, url=None):
        file_path = f"{Constants.COURSE_DIR}/home.html"
        content = FileUtils.get_content(file_path, encoding="utf-8")
        response = Selector(text=content)

        # mdd = response.xpath('//div[@class="list-item border-top"]/div//div/span/a').extract_first()
        # //*[@id="comments_list"]/div[2]/div/div[1]/div/div/div/div[1]/span
        bar_title_list = response.xpath('//div[@class="content-info"]/div[1]/span/text()').extract()

        course_info_list = self.extract_course_info_list(response=response)

        data = {
            "course_info_list": course_info_list,
        }

        logger.info(f'course_info_list:{len(course_info_list)}')

        city_bar_file_name = f"{self.out_json_dir}/course_info_list.json"
        FileUtils.save_to_json(city_bar_file_name, data)

        logger.info(f"success:{url} : {city_bar_file_name}")
        FileUtils.copy_file(city_bar_file_name, Constants.DIR_DATA_JSON_COURSE_INFO)

        success = True if len(course_info_list) > 0 else False
        return data, success

    def extract_course_info_list(self, response):
        """
        提取 course 数据
        :param response:
        :return:
        """
        el_info_list = response.xpath("//div[@class='content-info']").extract()
        info_list = []
        for index, info_bar in enumerate(el_info_list):
            info_content = Selector(text=info_bar)
            info_name = info_content.xpath('.//div[1]/span/text()').extract_first()
            info_type = info_content.xpath('.//div[2]/div[1]/text()').extract_first()
            info_time = info_content.xpath('.//div[2]/span[1]/text()').extract_first()
            info_count = info_content.xpath('.//div[2]/div[2]/text()').extract_first()
            info_status = info_content.xpath('.//div[2]/div[3]/text()').extract_first()

            info_count = str(info_count).replace("次学习", "")
            info_status = str(info_status).replace(" | ", "")

            raw_info = {
                "index": index,
                "info_name": info_name,
                "info_type": info_type,
                "info_time": info_time,
                "info_count": info_count,
                "info_status": info_status,
            }
            info = self.clean_info(raw_info)
            info_list.append(info)
        return info_list

    def clean_info(self, info):
        """
        清洗 info

        :param info:
        :return:
        """
        info_count = info["info_count"]
        info_status = info["info_status"]

        info_count = str(info_count).replace("次学习", "")
        if str(info_count).find("W") > -1:
            info_count = info_count.replace("W", "")
            info_count = float(info_count) * 10000

        info_count = int(info_count)
        info_status = str(info_status).replace(" | ", "")

        info["info_count"] = info_count
        info["info_status"] = info_status

        return info

    def build_header(self, url=None):
        headers = self.headers
        headers["Origin"] = self.base_url
        headers["cookie"] = self.user_cookies

        return headers

    def get_course_list(self, url):
        headers = self.build_header()

        for index in range(0, 14):
            data = f'bizData%5Bpage_size%5D={index}&bizData%5BisDesc%5D=1&bizData%5Bcontent_app_id%5D=&bizData%5Bchannel_id%5D=&bizData%5Bproduct_id%5D=p_5ef84e6ac5b04_zl7ToAc5&bizData%5Bqy_app_id%5D='

            info = self.sent_request(url, data=data, method='POST', header=headers, is_json=True)

            file_name = f"{self.out_json_dir}/course_list_{index}.json"
            FileUtils.save_to_json(file_name, content=info)
            logger.info(f"save file_name:{file_name}")

    def get_course_list_v2(self, ):
        """
            获取该课程下的每节课的基本信息：app_id，resource_id，title等

        :return:返回课程列表,主要是获取其中的课程对应id信息
        """
        # last_id不要设置，order_weight按课程总数设置起始与终止数
        data = {
            'bizData[resource_id]': self.resource_id,
            'bizData[product_id]': self.product_id,
            'bizData[content_app_id]': '',
            'bizData[qy_app_id]': '',
            'bizData[page_num]': '140',
            'bizData[last_id]': '',
            'bizData[start_order_weight]': '0',
            'bizData[end_order_weight]': ''
        }

        info, save_file_name = self.send_post(url=self.course_list_v2_url, data=data, file_name="course_list_v2.json")
        return info

    def get_course_video_info(self, resource_id=None, course=None, course_index=None):
        """
        get_course_video_info

        :param resource_id:
        :param course:
        :param course_index:
        :return:
        """

        pay_info = {
            "type": "2",
            "product_id": self.product_id,
            "from_multi_course": "1",
            "resource_id": resource_id,
            "resource_type": 3,
            "app_id": self.app_id,
            "payment_type": ""
        }
        data = {'pay_info': json.dumps(pay_info)}

        # data = f"pay_info=%7B%22type%22%3A%222%22%2C%22product_id%22%3A%22p_5ef84e6ac5b04_zl7ToAc5%22%2C%22from_multi_course%22%3A%221%22%2C%22resource_id%22%3A%22{resource_id}%22%2C%22resource_type%22%3A3%2C%22app_id%22%3A%22appiXguJDJJ6027%22%2C%22payment_type%22%3A%22%22%7D"
        course_index = course["index"] if course_index is None else course_index
        file_name = f"course/course_info_{course_index}.json"
        info, save_file_name = self.send_post(url=self.video_info_url, data=data, file_name=file_name)
        return info, save_file_name

    def get_column_info(self):
        """
        获取栏目信息
        :return:
        """
        data = 'pay_info=%7B%22type%22%3A%223%22%2C%22source%22%3A%222%22%2C%22content_app_id%22%3A%22%22%2C%22qy_app_id%22%3A%22%22%2C%22product_id%22%3A%22p_5ef84e6ac5b04_zl7ToAc5%22%2C%22resource_type%22%3A6%2C%22app_id%22%3A%22appiXguJDJJ6027%22%2C%22payment_type%22%3A%22%22%2C%22resource_id%22%3A%22%22%7D'
        info, save_file_name = self.send_post(url=self.column_info_url, data=data, file_name="column_info.json")
        return info

    def send_post(self, url, data, file_name, method='POST', is_json=True):
        """
        send post to get json result

        :param url:
        :param data:
        :param file_name:
        :param method:
        :param is_json:
        :return:
        """
        save_file_name = f"{self.out_json_dir}/{file_name}"
        if not self.retry and FileUtils.check_file_exists(save_file_name):
            info = FileUtils.load_to_json(save_file_name)
            logger.info(f"load json from file_name:{save_file_name}")
            return info, save_file_name

        headers = self.build_header()
        info = self.sent_request(url=url, data=data, method=method, header=headers, is_json=is_json)

        FileUtils.save_to_json(save_file_name, content=info)
        logger.info(f"save file_name:{save_file_name}")
        return info, save_file_name

    def merge_video_info(self):
        """
        merge vido info
        :return:
        """
        course_info_list_file_name = f"{self.out_json_dir}/course_info_list.json"
        all_video = FileUtils.load_to_json(course_info_list_file_name)
        video_title_mapping = {item["info_name"]: item["index"] for item in all_video["course_info_list"]}
        for index in range(0, self.total_page + 1):
            all_video = FileUtils.load_to_json(course_info_list_file_name)
            file_name = f"{self.out_json_dir}/course_list_{index}.json"
            course_list = FileUtils.load_to_json(file_name)
            # logger.info(f"{index} - {course_list}")

            course_info_list = course_list["data"]["contentData"]["contentInfo"]
            for idx, course_info in enumerate(course_info_list):
                title = course_info["title"]
                video_index = video_title_mapping.get(title, -1)
                if video_index < 0:
                    logger.info(f"not find: {idx} - {title} - {course_info}")
                    continue
                all_video["course_info_list"][video_index]["info"] = course_info

            FileUtils.save_to_json(course_info_list_file_name, all_video)

            logger.info(f"process one page: {index} - {len(course_info_list)}")

    def get_course_list_video_info(self):
        """
        get course list video info

        :return:
        """
        course_info_list_file_name = f"{self.out_json_dir}/course_info_list.json"
        all_video = FileUtils.load_to_json(course_info_list_file_name)
        video_title_mapping = {item["info_name"]: item["index"] for item in all_video["course_info_list"]}
        course_info_list = all_video["course_info_list"]

        logger.info(f"total video list :{len(course_info_list)}")
        for index, course in enumerate(course_info_list):
            info_name = course["info_name"]
            info_type = course["info_type"]
            resource_id = course["info"]["resource_id"]
            if info_type in ["图文", "直播"]:
                logger.info(f"{index} - {info_name} is {info_type}")
                continue
            # if "video_info" in course:
            #     logger.info(f"{index} - {info_name} have down")
            #     continue

            logger.info(f"{index} - {info_name} begin ")
            video_info, video_info_file_name = self.get_course_video_info(resource_id=resource_id, course=course)

            course["video_info"] = video_info["data"]["bizData"]
            FileUtils.save_to_json(course_info_list_file_name, all_video)
            logger.info(f"process one course: {index} - {info_name}")
            self.sleep(max_second=5)

    def download_audio(self):
        course_info_list_file_name = f"{self.out_json_dir}/course_info_list.json"
        all_video = FileUtils.load_to_json(course_info_list_file_name)
        video_title_mapping = {item["info_name"]: item["index"] for item in all_video["course_info_list"]}
        course_info_list = all_video["course_info_list"]

        logger.info(f"total video list :{len(course_info_list)}")
        save_file_type = "audio"
        thread_size = 10
        audio_url_list = []
        for index, course in enumerate(course_info_list):
            info_index = course["index"]
            info_name = course["info_name"]
            info_type = course["info_type"]
            resource_id = course["info"]["resource_id"]

            if info_type in ["图文", "直播"]:
                logger.info(f"{index} - {info_name} is {info_type}")
                continue

            video_info = course["video_info"]["data"]
            video_audio_url = video_info["video_audio_url"]

            video_audio_save_name = FileUtils.get_save_file_name_from_url(save_dir=self.out_audio_dir,
                                                                          url=video_audio_url,
                                                                          save_file_type=save_file_type)
            video_audio_info = {
                "index": info_index,
                "info_name": info_name,
                "resource_id": resource_id,
                "url": video_audio_url,
                "save_name": video_audio_save_name,
            }
            audio_url_list.append(video_audio_info)
            logger.info(f"{index} - {info_name} have down")

            course["video_audio_url"] = video_audio_url
            course["video_save_name"] = video_audio_save_name

            FileUtils.save_to_json(course_info_list_file_name, all_video)

        save_video_audio_info_file_name = f"{self.out_json_dir}/course_video_audio_list.json"
        FileUtils.save_to_json(save_video_audio_info_file_name, content=audio_url_list)
        logger.info(f"save file_name:{save_video_audio_info_file_name}")

        logger.info(f"begin to download audio: {len(audio_url_list)}")
        url_list = [item['url'] for item in audio_url_list]
        download_v2(url_list=url_list, file_path=Constants.COURSE_AUDIO_DIR,
                    thread_size=thread_size, save_file_type=save_file_type)

    def download_video_url(self):
        """
        获取视频的url
        :return:
        """
        course_info_list_file_name = f"{self.out_json_dir}/course_info_list.json"
        all_video = FileUtils.load_to_json(course_info_list_file_name)
        video_title_mapping = {item["info_name"]: item["index"] for item in all_video["course_info_list"]}
        course_info_list = all_video["course_info_list"]

        logger.info(f"total video list :{len(course_info_list)}")

        course_video_list_file_name = f"{self.out_json_dir}/course_video_list.json"
        video_url_list = []
        for index, course in enumerate(course_info_list):
            course_index = course["index"]
            title = course["info_name"]
            info_type = course["info_type"]
            resource_id = course["info"]["resource_id"]

            if info_type in ["图文", "直播"]:
                logger.info(f"{index} - {title} is {info_type}")
                continue

            # if "video_download" in course:
            #     logger.info(f"{index} - {title} is down")
            #     continue

            if course_index <= 3:
                continue

            video_info = course["video_info"]["data"]
            video_url = video_info["videoUrl"]
            video_full_path = f"{self.out_video_dir}/{course_index}_{title}.mp4"
            video_m3u8_path = f"{video_full_path[:-4]}.m3u8"
            video_key_path = f"{video_full_path[:-4]}.key"

            # if FileUtils.check_file_exists(video_m3u8_path) and FileUtils.check_file_exists(video_key_path):
            #     logger.info(f"{index} - {title} have down")
            #     continue

            # 解码video url and download m3u8 ,key file
            video_info_m3u8 = self.decode_video_url(video_encode_url=video_url,
                                                    course_index=course_index,
                                                    title=title)
            # download
            logger.info(f"正在下载:{title}......")

            video_info_m3u8["video_full_path"] = video_full_path
            video_info_m3u8["video_ts_path"] = f"{video_full_path[:-4]}.ts"
            video_info_m3u8["video_m3u8_path"] = video_m3u8_path
            video_info_m3u8["video_key_path"] = video_key_path

            key, key_file_name = self.get_video_key(video_info_m3u8)

            video_info_m3u8["resource_id"] = resource_id
            video_info_m3u8["origin_video_url"] = video_url

            course["video_download"] = video_info_m3u8

            video_url_list.append(video_info_m3u8)
            logger.info(f"{index} - {title} - {video_full_path} have down")
            FileUtils.save_to_json(course_info_list_file_name, all_video)
            FileUtils.save_to_json(course_video_list_file_name, video_url_list, )

            self.sleep(max_second=5)

        FileUtils.save_to_json(course_video_list_file_name, video_url_list, )
        logger.info(f"save video url list: {course_video_list_file_name}")

    def download_video(self):
        """
        获取视频的url
        :return:
        """
        course_info_list_file_name = f"{self.out_json_dir}/course_info_list.json"
        all_video = FileUtils.load_to_json(course_info_list_file_name)
        video_title_mapping = {item["info_name"]: item["index"] for item in all_video["course_info_list"]}
        course_info_list = all_video["course_info_list"]

        for index, course in enumerate(course_info_list):
            course_index = course["index"]
            title = course["info_name"]
            info_type = course["info_type"]
            resource_id = course["info"]["resource_id"]

            if info_type in ["图文", "直播"]:
                logger.info(f"{index} - {title} is {info_type}")
                continue

            if "video_download" not in course:
                logger.info(f"{index} - {title} video_download not in course, skip")
                continue

            if course_index <= 1:
                continue

            video_info_m3u8 = course["video_download"]
            video_url = video_info_m3u8["video_url"]
            video_full_path = video_info_m3u8["video_full_path"]
            video_ts_path = video_info_m3u8["video_ts_path"]

            # if FileUtils.check_file_exists(video_ts_path) or FileUtils.check_file_exists(video_full_path):
            #     logger.info(f"{index} - {title} have download down")
            #     continue
            logger.info(f"begin to download: {index} - {title} - {video_url} - {video_ts_path}")
            self.download_file(file_url=video_url, file_name=video_ts_path)

            # 请求视频内容，并解密保存
            # video = requests.get(video_url, headers=self.headers, stream=True)
            self.decode_video(video_info=video_info_m3u8)

            logger.info(f"下载完成: {index} - {title} - {video_full_path}")

            self.sleep(max_second=5)

    def download_video_multi_thread(self):
        course_video_list_file_name = f"{self.out_json_dir}/course_video_list.json"
        course_video_list = FileUtils.load_to_json(course_video_list_file_name)
        need_web_url_file_name = f"{self.out_json_dir}/course_need_web_man_list.json"
        modify_need_web_url_file_name = f"{Constants.DIR_DATA_JSON_COURSE_INFO}/course_need_web_man_list.json"
        FileUtils.copy_file(modify_need_web_url_file_name, self.out_json_dir)

        need_web_url_video_list = FileUtils.load_to_json(modify_need_web_url_file_name)
        need_web_url_mapping = {item["resource_id"]: index for index, item in enumerate(need_web_url_video_list)}

        all_video_url_list = []
        for index, course_video in enumerate(course_video_list):
            video_url = course_video["video_url"]
            # video_url_web = course_video.get("video_url_web", None)
            video_url_web = None
            video_index = course_video["index"]
            title = course_video["title"]
            video_ts_path = course_video["video_ts_path"]
            resource_id = course_video["resource_id"]

            if resource_id in need_web_url_mapping.keys():
                need_index = need_web_url_mapping[resource_id]
                raw_video_url_web = need_web_url_video_list[need_index]["url"]
                if len(raw_video_url_web) < 2:
                    video_url_web = None
                else:
                    video_url_web = self.modify_video_url(raw_video_url_web)

            if video_index <= 71 or video_index >= 148:
                logger.info(f"{video_index} - {title} has down ")
                continue

            video_save_path = f"{self.out_video_dir_big}/{video_index}_{title}.ts"
            if video_url_web is None and FileUtils.check_file_exists(video_save_path):
                logger.info(f"{index} - {title} - {video_save_path} have download down")
                continue

            if video_url_web is None:
                logger.info(f"{index} - {title} - {video_save_path} is none")
                continue

            download_video_url = video_url if video_url_web is None else video_url_web
            video_info = {
                "url": download_video_url,
                "file": video_save_path,
            }
            if video_url_web is not None:
                all_video_url_list.append(video_info)

        logger.info(f"need total download: {len(all_video_url_list)}")
        logger.info(f"save dir: {self.out_video_dir_big}")

        for item in all_video_url_list:
            logger.info(f"need download: {item['file']} - {item['url']} - ")

        thread_size = 5
        sleep_max_second = 10
        multi_downloader = MultiTheadDownloader(file_url_list=all_video_url_list,
                                                save_dir=self.out_video_dir_big,
                                                thread_size=thread_size, save_file_type=None,
                                                sleep_max_second=sleep_max_second)

        multi_downloader.start()
        multi_downloader.stop_treads()

    def merge_all_video(self):
        course_video_list_file_name = f"{self.out_json_dir}/course_video_list.json"
        course_video_list = FileUtils.load_to_json(course_video_list_file_name)

        all_video_url_list = []
        for index, course_video in enumerate(course_video_list):
            video_url = course_video["video_url"]
            video_index = course_video["index"]
            title = course_video["title"]
            video_ts_path = course_video["video_ts_path"]
            video_key_path = course_video["video_key_path"]

            if video_index <= 3:
                logger.info(f"{video_index} - {title} has down ")
                continue

            video_save_path = f"{self.out_video_dir_big}/{video_index}_{title}.ts"
            video_full_path = f"{video_save_path[:-3]}.mp4"
            if not FileUtils.check_file_exists(video_save_path) or FileUtils.check_file_exists(video_full_path):
                logger.info(f"{index} - {title} - {video_save_path} have not download down")
                continue

            video_info_m3u8 = {
                "video_key_path": video_key_path,
                "video_ts_path": video_save_path,
                "video_full_path": video_full_path,
            }
            self.decode_video(video_info=video_info_m3u8)

            logger.info(f"{index} - {title} - {video_save_path} have  decode_video down")

    def build_m3u8_file_for_ffmpeg(self, video_info):

        m3u8_url = video_info["m3u8_url"]
        video_url = video_info["video_url"]
        m3u8_file_name = video_info["video_m3u8_path"]
        key_file_name = video_info["video_key_path"]

        new_m3u8_file_name = f"{m3u8_file_name[:-5]}_1.m3u8"

        m3u8_content = FileUtils.read_to_text_list(file_name=m3u8_file_name)

        key_url = None
        new_m3u8_content = []

        video_url_prefix = video_url.split('drm')[0]

        # "v.f421220_0.ts?" "drm/v.f421220.ts?"
        ts_begin = f"{self.video_url_prefix[4:-4]}_0.ts?"
        for idx, content in enumerate(m3u8_content):
            ts_pos = content.find(ts_begin)
            new_content = str(content).strip("\n")
            if ts_pos > -1:
                new_content = video_url_prefix + content

            new_m3u8_content.append(new_content)

        FileUtils.save_to_text(new_m3u8_file_name, "\n".join(new_m3u8_content))
        logger.info(f"new_m3u8_file_name:{new_m3u8_file_name}")

    def download_video_use_ffmpeg(self):
        course_video_list_file_name = f"{self.out_json_dir}/course_video_list.json"
        course_web_video_list_file_name = f"{self.out_json_dir}/course_web_video_list.json"
        course_video_list = FileUtils.load_to_json(course_video_list_file_name)
        course_web_ts_video_list = FileUtils.read_to_text_list(self.out_web_video_dir)

        all_video_url_list = []
        need_web_url_list = []
        run_index = 0
        for index, course_video in enumerate(course_video_list):
            video_url = course_video["video_url"]
            video_index = course_video["index"]
            resource_id = course_video["resource_id"]
            title = course_video["title"]
            video_ts_path = course_video["video_ts_path"]
            video_m3u8_path = course_video["video_m3u8_path"]

            if video_index <= 3:
                logger.info(f"{video_index} - {title} has down ")
                continue

            video_save_path = f"{self.out_video_dir_big}/{video_index}_{title}.ts"
            if FileUtils.check_file_exists(video_save_path):
                logger.info(f"{index} - {title} - {video_save_path} have download down")
                continue

            course_web_url = f"{self.course_web_url}/{resource_id}?type=2&pro_id={self.product_id}&from_multi_course=1"
            logger.info(f"{video_index} - {title} - {course_web_url}")

            course_video["course_web_url"] = course_web_url

            if run_index < 12:
                course_web_ts_video_url = course_web_ts_video_list[run_index]
                pre_pos = 87
                video_url_pre = video_url[:pre_pos]
                course_web_ts_video_url_pre = course_web_ts_video_url[:pre_pos]
                if video_url_pre != course_web_ts_video_url_pre:
                    logger.info(f"视频链接不一致: {video_index} - {title}")
                    logger.info(f"url1:{video_url}")
                    logger.info(f"url2:{course_web_ts_video_url}")
                    continue
                video_url_web = self.modify_video_url(course_web_ts_video_url)
                course_video["video_url_web"] = video_url_web
                run_index += 1

            FileUtils.save_to_json(course_video_list_file_name, course_video_list)

            need_web_url_list.append(deepcopy(course_video))

            # FileUtils.read_to_text_list(video_ts_path)
        FileUtils.save_to_json(course_web_video_list_file_name, need_web_url_list)
        logger.info(f"course_web_video_list_file_name:{course_web_video_list_file_name}")
        FileUtils.copy_file(course_web_video_list_file_name, Constants.DIR_DATA_JSON_COURSE_INFO)
        FileUtils.copy_file(course_video_list_file_name, Constants.DIR_DATA_JSON_COURSE_INFO)

    def modify_video_url(self, course_web_ts_video_url):
        """
        提取单个视频下载链接
        :param course_web_ts_video_url:
        :return:
        """
        video_url1 = course_web_ts_video_url.split("start=")[0]
        video_url2 = "type=" + course_web_ts_video_url.split("type=")[1]
        video_url_web = video_url1 + video_url2
        return video_url_web


def demo_extract_home():
    course_spider = CourseSpider()
    # course_spider.get_home_list()
    # course_spider.get_course_video_info()
    # course_spider.get_course_list()
    # course_spider.get_course_list_v2()
    # course_spider.merge_video_info()
    # course_spider.get_column_info()
    # course_spider.get_course_list_video_info()
    # course_spider.download_audio()
    # course_spider.get_course_video_info_v2()
    # course_spider.download_video_url()
    # course_spider.download_video()
    # course_spider.download_video_multi_thread()
    # course_spider.merge_all_video()
    # course_spider.download_video_use_ffmpeg()


if __name__ == '__main__':
    demo_extract_home()
