#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project ml-learn
# @File  : test_spider_course.py
# @Author: sl
# @Date  : 2022/5/22 - 上午11:22
import unittest

from util.logger_utils import logger
from util.v2.constants import Constants
from util.v2.file_utils import FileUtils


class TestSpiderCourse(unittest.TestCase):

    def setUp(self) -> None:
        # super().__init__()
        self.config = FileUtils.load_to_json(filename=Constants.COURSE_CONFIG_FILE)
        self.config_run_args = self.config['run_args']
        self.out_dir = self.config_run_args["out_dir"]
        self.out_json_dir = self.config_run_args["out_json_dir"]
        self.out_audio_dir = self.config_run_args["out_audio_dir"]
        self.out_video_dir = self.config_run_args["out_video_dir"]
        self.out_video_dir_big = self.config_run_args["out_video_dir_big"]
        self.out_web_video_dir = self.config_run_args["out_web_video_dir"]

    def test_video_web_url(self):
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
                logger.info(f"{video_index} - {title}  - {video_save_path} have download down")
                continue

            if run_index < 11:
                run_index += 1
            # course_web_url = f"{self.course_web_url}/{resource_id}?type=2&pro_id={self.product_id}&from_multi_course=1"
            course_web_url = course_video["course_web_url"]
            logger.info(f"{video_index} - {title} - {course_web_url}")

            course_video["url"] = ""

            need_web_url_list.append(course_video)

        need_web_url_file_name = f"{self.out_json_dir}/course_need_web_man_list.json"
        FileUtils.save_to_json(need_web_url_file_name, need_web_url_list)
        FileUtils.copy_file(need_web_url_file_name, Constants.DIR_DATA_JSON_COURSE_INFO)
        logger.info(f"need_web_url_file_name:{need_web_url_file_name}")
