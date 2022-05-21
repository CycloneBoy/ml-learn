#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project ml-learn
# @File  : parse_home.py
# @Author: sl
# @Date  : 2022/5/20 - 下午8:17
import random
import time
from urllib import parse

import requests
from scrapy import Selector

from basis.spider.download_v1 import download_v2
from basis.spider.mafengwo.extract_base import ExtractSpiderBase
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
        self.total_page = self.config_spider['total_page']

        self.config_run_args = self.config['run_args']
        self.out_dir = self.config_run_args["out_dir"]
        self.out_json_dir = self.config_run_args["out_json_dir"]
        self.out_audio_dir = self.config_run_args["out_audio_dir"]
        self.out_video_dir = self.config_run_args["out_video_dir"]

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

    def get_course_video_info(self, resource_id=None, course=None):
        """
        get_course_video_info
        :param resource_id:
        :param course:
        :return:
        """
        data = f"pay_info=%7B%22type%22%3A%222%22%2C%22product_id%22%3A%22p_5ef84e6ac5b04_zl7ToAc5%22%2C%22from_multi_course%22%3A%221%22%2C%22resource_id%22%3A%22{resource_id}%22%2C%22resource_type%22%3A3%2C%22app_id%22%3A%22appiXguJDJJ6027%22%2C%22payment_type%22%3A%22%22%7D"
        course_index = course["index"]
        info, save_file_name = self.send_post(url=self.video_info_url, data=data,
                                              file_name=f"course/course_info_{course_index}.json")
        return info, save_file_name

    def get_column_info(self):
        """
        获取栏目信息
        :return:
        """
        data = 'pay_info=%7B%22type%22%3A%223%22%2C%22source%22%3A%222%22%2C%22content_app_id%22%3A%22%22%2C%22qy_app_id%22%3A%22%22%2C%22product_id%22%3A%22p_5ef84e6ac5b04_zl7ToAc5%22%2C%22resource_type%22%3A6%2C%22app_id%22%3A%22appiXguJDJJ6027%22%2C%22payment_type%22%3A%22%22%2C%22resource_id%22%3A%22%22%7D'
        info, save_file_name = self.send_post(url=self.column_info_url, data=data, file_name="column_info.json")
        return info

    def send_post(self, url, data, file_name):
        """
        send post to get json result

        :param url:
        :param data:
        :param file_name:
        :return:
        """
        headers = self.build_header()
        info = self.sent_request(url=url, data=data, method='POST', header=headers, is_json=True)

        save_file_name = f"{self.out_json_dir}/{file_name}"
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
            if "video_info" in course:
                logger.info(f"{index} - {info_name} have down")
                continue

            logger.info(f"{index} - {info_name} begin ")
            video_info, video_info_file_name = self.get_course_video_info(resource_id=resource_id, course=course)

            course["video_info"] = video_info["data"]["bizData"]
            FileUtils.save_to_json(course_info_list_file_name, all_video)
            logger.info(f"process one course: {index} - {info_name}")
            sleep_time = random.random() * 5
            logger.info(f"sleep_time:{sleep_time}")
            time.sleep(sleep_time)

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

    def download_video(self):
        course_info_list_file_name = f"{self.out_json_dir}/course_info_list.json"
        all_video = FileUtils.load_to_json(course_info_list_file_name)
        video_title_mapping = {item["info_name"]: item["index"] for item in all_video["course_info_list"]}
        course_info_list = all_video["course_info_list"]

        logger.info(f"total video list :{len(course_info_list)}")

        video_url_list = []
        for index, course in enumerate(course_info_list):
            info_index = course["index"]
            info_name = course["info_name"]
            info_type = course["info_type"]
            resource_id = course["info"]["resource_id"]

            if info_type in ["图文", "直播"]:
                logger.info(f"{index} - {info_name} is {info_type}")
                continue

            video_info = course["video_info"]["data"]
            video_url = video_info["videoUrl"]

            video_simple_info = {
                "index": info_index,
                "info_name": info_name,
                "resource_id": resource_id,
                "url": video_url,
            }
            video_url_list.append(video_simple_info)
            logger.info(f"{index} - {info_name} - {video_url}")
            logger.info(f"{index} - {info_name} have down")


def demo_extract_home():
    course_spider = CourseSpider()
    # course_spider.get_home_list()
    # course_spider.get_course_video_info()
    # course_spider.get_course_list()
    course_spider.get_course_list_v2()
    # course_spider.merge_video_info()
    # course_spider.get_column_info()
    # course_spider.get_course_list_video_info()
    # course_spider.download_audio()
    # course_spider.download_video()


if __name__ == '__main__':
    demo_extract_home()
