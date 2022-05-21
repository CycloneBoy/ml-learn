#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : file_utils.py
# @Author: sl
# @Date  : 2022/2/18 - 下午10:48
import shutil
from urllib.parse import urlparse

from util.v2.common_utils import BaseUtils

import glob
import json
import os
import pickle
import random

import pandas as pd

from util.constant import DATA_TXT_NEWS_DIR, DATA_TXT_STOP_WORDS_GITHUB_DIR, DATA_HTML_DIR, \
    DATA_QUESTION_ANSWER_DIR, DATA_CACHE_DIR
from util.logger_utils import get_log
from util.nlp_utils import extract_chinese
from util.v2.constants import Constants

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))


class FileUtils(BaseUtils):
    """
    file utils
    """

    @staticmethod
    def get_content(path, encoding='gbk'):
        """
        读取文本内容
        :param path:
        :param encoding:
        :return:
        """
        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            content = ''
            for l in f:
                l = l.strip()
                content += l
            return content

    @staticmethod
    def get_news_path(sub_path='C000008'):
        """
        获取新闻预料的路径
        :param sub_path:
        :return:
        """
        return os.path.join(DATA_TXT_NEWS_DIR, sub_path)

    @staticmethod
    def build_stop_words(path=DATA_TXT_STOP_WORDS_GITHUB_DIR,
                         save_file_name=os.path.join(DATA_TXT_STOP_WORDS_GITHUB_DIR, 'stop_words.utf8')):
        files = glob.glob(os.path.join(path, '*.txt'))
        words = []
        for file in files:
            with open(file, 'r', errors='ignore') as f:
                for line in f.readlines():
                    words.append(line)

        print("文件长度: %d 停用词长度:%d " % (len(files), len(words)))
        print("样例:%s " % words[100])

        result = []
        filter_set = set()
        for word in words:
            if word not in filter_set:
                result.append(word)
                filter_set.add(word)

        with open(save_file_name, 'w', errors='ignore') as f:
            f.writelines(result)
        print("保存处理后的停用词字典: %s ,停用词长度:%d " % (save_file_name, len(result)))

    @staticmethod
    def test_get_one_news():
        files = glob.glob(os.path.join(FileUtils.get_news_path(), '*.txt'))
        corpus = [FileUtils.get_content(file) for file in files]

        sample_inx = random.randint(0, len(corpus))
        print(corpus[sample_inx])

    @staticmethod
    def save_to_text(filename, content, mode='w'):
        """
        保存为文本
        :param filename:
        :param content:
        :return:
        """
        FileUtils.check_file_exists(filename)
        with open(filename, mode, encoding='utf-8') as f:
            f.writelines(content)

    @staticmethod
    def save_to_json(filename, content):
        """
        保存map 数据
        :param filename:
        :param maps:
        :return:
        """
        FileUtils.check_file_exists(filename)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)

    @staticmethod
    def load_to_json(filename):
        """
        加载 数据
        :param filename:
        :param maps:
        :return:
        """
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def get_file_name_list(file_name, type="*.txt"):
        """获取指定路径下的指定类型的所有文件"""
        files = glob.glob(os.path.join(file_name, type))
        return files

    @staticmethod
    def check_file_exists(filename, delete=False):
        """检查文件是否存在"""
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            log.info("文件夹不存在,创建目录:{}".format(dir_name))
        return os.path.exists(filename)

    @staticmethod
    def read_to_text(file_name, encoding='utf-8'):
        """读取txt 文件"""
        with open(file_name, 'r', encoding=encoding) as f:
            content = f.read()
            return content

    @staticmethod
    def read_to_text_list(file_name, encoding='utf-8'):
        """
        读取txt文件,默认utf8格式,
        :param path:
        :param encoding:
        :return:
        """
        list_line = []
        with open(file_name, 'r', encoding=encoding) as f:
            list_line = f.readlines()
            return list_line

    @staticmethod
    def list_file(file_dir, endswith=""):
        """读取文件列表"""
        file_list = []
        if not os.path.exists(file_dir):
            return file_list
        for file in os.listdir(file_dir):
            if file.endswith(endswith):
                file_list.append(file)

        return file_list

    @staticmethod
    def delete_file(file_name):
        """
        删除一个目录下的所有文件
        :param file_name:
        :return:
        """

        for i in os.listdir(file_name):
            path_children = os.path.join(file_name, i)
            if os.path.isfile(path_children):
                os.remove(path_children)
            else:  # 递归, 删除目录下的所有文件
                FileUtils.delete_file(path_children)

    @staticmethod
    def read_and_process(path):
        """
          读取文本数据并
        :param path:
        :return:
        """
        data = pd.read_csv(path)
        ques = data["ques"].values.tolist()
        labels = data["label"].values.tolist()
        line_x = [extract_chinese(str(line).upper()) for line in labels]
        line_y = [extract_chinese(str(line).upper()) for line in ques]
        return line_x, line_y

    @staticmethod
    def build_qa_dataset(file_dir):
        """读取问答数据集"""
        file_list = FileUtils.list_file(file_dir, ".txt")

        filename = os.path.join(DATA_CACHE_DIR, "question/travel_question_63752.txt")
        total = 0
        for file in file_list:
            total_line = int(file[file.find("_") + 1:file.find(".")])
            total += total_line
            path = os.path.join(file_dir, file)
            contents = FileUtils.read_to_text(path)
            FileUtils.save_to_text(filename, contents, 'a')

        log.info("文件数：{}，总共问题数量：{}".format(len(file_list), total))

    @staticmethod
    def get_path_dir(file_dir):
        """
        读取文件夹下的所有子文件夹
        :param file_dir:
        :return:
        """
        dir_list = []
        # files_list = []
        for root, dirs, files in os.walk(file_dir):
            dir_list.extend(dirs)
            # files_list.extend(files)
        return dir_list

    @staticmethod
    def save_to_pickle(model, file_name):
        """
        保存模型
        :param model:
        :param file_name:
        :return:
        """
        FileUtils.check_file_exists(file_name)
        pickle.dump(model, open(file_name, "wb"))

    @staticmethod
    def load_to_model(file_name):
        """
         使用pickle加载模型文件
        :param file_name:
        :return:
        """
        loaded_model = pickle.load(open(file_name, "rb"))
        return loaded_model

    @staticmethod
    def get_dir(file_name):
        """
        获取文件的目录
        :param file_name:
        :return:
        """
        return os.path.dirname(os.path.abspath(file_name))

    @staticmethod
    def get_file_name(file_name):
        """
        获取文件的名称
        :param file_name:
        :return:
        """

        file_dir = os.path.abspath(file_name)
        name = file_dir[file_dir.rindex('/') + 1:]
        return name

    @staticmethod
    def get_url_file_name(url):
        """
        获取文件的名称
        :param url:
        :return:
        """

        file_dir = FileUtils.get_url_file_path(url, make_dir=False)
        name = os.path.basename(file_dir)
        name = name[:name.rindex('.')]
        return name

    @staticmethod
    def get_url_file_parent_name(url):
        """
        获取文件的父目录名称
        :param url:
        :return:
        """
        parse_url = urlparse(url)
        url_path = parse_url.path
        file_dir = os.path.dirname(url_path)
        parent_dir_name = os.path.basename(file_dir)
        return parent_dir_name

    @staticmethod
    def get_album_photo_path(url):
        """
        获取游记图片详情 save path
           http://www.mafengwo.cn/photo/18671/scenery_23513200/1694284233.html

        :param url:
        :return:
        """
        travel_id = FileUtils.get_url_file_parent_name(url)
        travel_id = travel_id.split("_")[1]
        image_id = FileUtils.get_url_file_name(url)

        file_path = FileUtils.get_url_file_path(url)
        save_dir = os.path.dirname(file_path)
        save_file_name = f"{save_dir}/{travel_id}_{image_id}.json"

        return save_file_name

    @staticmethod
    def get_url_file_path(url, base_dir=Constants.SPIDER_MAFENGWO_DIR, make_dir=True):
        """
        获取文件的名称
        :param url:   "http://www.mafengwo.cn/travel-scenic-spot/mafengwo/12711.html"
                        "/photo/mdd/12711.html"
        :param base_dir:
        :param make_dir:
        :return:
        """
        parse_url = urlparse(url)

        name = f"{base_dir}{parse_url.path}"
        file_dir = os.path.dirname(name)
        if make_dir:
            os.makedirs(file_dir, exist_ok=True)
        return name

    @staticmethod
    def copy_file(src, dst):
        """
        获取文件的名称
        :param src:
        :param dst:
        :return:
        """
        if not os.path.exists(dst):
            dir_name = os.path.dirname(dst)
            os.makedirs(dir_name, exist_ok=True)
        return shutil.copy(src=src, dst=dst)

    @staticmethod
    def get_save_file_name_from_url(save_dir, url, save_file_type=None):
        """
        获取文件的保存路径
        :param save_dir: 
        :param url: 
        :param save_file_type: 
        :return: 
        """
        sp_filename = url.split("/")
        if save_file_type == "audio":
            filename = f"{save_dir}/{sp_filename[len(sp_filename) - 2]}.mp3"
        else:
            filename = save_dir + "/" + sp_filename[len(sp_filename) - 1]

        return filename


if __name__ == '__main__':
    pass
