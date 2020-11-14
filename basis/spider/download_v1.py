#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : download_v1.py
# @Author: sl
# @Date  : 2020/11/8 - 下午10:26

import json
import os
import urllib3
from lxml import etree
from scrapy import Selector

from basis.utils.multi_thread_download import multi_download, stop_treads, log, setImageDir

filename = "course/data2.json"
file_index_html = "/home/sl/workspace/python/a2020/ml-learn/basis/spider/course/index.html"

http = urllib3.PoolManager()


def read_data(filename):
    video_list = []
    # 读取数据
    with open(filename, 'r') as f:
        response = json.load(f)
        # log.info(response)

        list = response['data']['resource_list']
        log.info(list)

        for key in list.keys():
            video = {}
            value = list[key]
            # log.info("{} ->{}".format(key, value))

            target_name = ''
            if 'file_name' in value:
                target_name = value['file_name']
                video['file_name'] = target_name
            if 'transcoding_ext' in value:
                transcoding_ext = value['transcoding_ext']
                if transcoding_ext == 'mp4':
                    video['url'] = value['url']
                    video['transcoding_url'] = value['transcoding_url']
                video_list.append(video)
    return video_list


def download(url, file_name, file_path='/home/sl/workspace/data/test'):
    r = http.request('GET', url)
    save_path = os.path.join(file_path, file_name)
    with open(save_path, "wb") as code:
        code.write(r.data)


# 下载文件
def download_v2(url_list, file_path="/home/sl/workspace/python/mafengwo/imagehome/video", thread_size=10):
    threads, start_time = multi_download(url_list, image_dir=file_path, thread_size=thread_size)
    stop_treads(threads, start_time, url_list)


def rename_file(file_path, name_dic):
    fileList = os.listdir(file_path)
    for name in fileList:
        old_name = os.path.join(file_path, name)
        # log.info(old_name)
        if name in name_dic:
            new_name = os.path.join(file_path, name_dic[name])
            os.rename(old_name, new_name)
            log.info("文件重命名:{} 为:{} ".format(old_name, new_name))


# 根据url 获取文件的名称
def get_file_name(file_url):
    file = str(file_url).split('/')
    name = file[len(file) - 1]
    return name


def read_html(filename):
    content = None
    with open(filename, 'r') as f:
        content = f.read()
    return content


# 根据请求的JSON 来下载视频
def download_video_from_json():
    video_list = read_data(filename)
    url_list = []
    name_dic = {}
    for video in video_list:
        file_url = video['url']
        file_name = video['file_name']
        name = get_file_name(file_url)
        log.info("{} -> {}".format(file_name, file_url))
        url_list.append(file_url)
        name_dic[name] = file_name
    download_file_and_rename(url_list, name_dic)


# 下载文件并重命名文件
def download_file_and_rename(url_list, name_dic=None,file_path="/home/sl/workspace/python/mafengwo/imagehome/video", thread_size=10):
    download_v2(url_list,file_path=file_path,thread_size=thread_size)

    if name_dic != None:
    # file_path = os.path.join(setImageDir, 'video')
        rename_file(file_path, name_dic)


# 获取视频的名称列表
def get_video_name_list(html):
    title_list = html.xpath('//div[contains(@class,"session-title")]/div/span[@class="title"]/text()').extract()
    log.info("video title list length:{}".format(len(title_list)))
    name_list = []
    for index, name in enumerate(title_list):
        video_name = "{}.{}.mp4".format(index+1, name).replace(" ","")
        name_list.append(video_name)
        # log.info(video_name)
    # 去除收尾 非视频名称
    return name_list[1:-1]


# 获取每个视频的时间长度
def get_video_time_list(html):
    video_time_list = html.xpath('//div[contains(@class,"session-banner ")]/div[@class="time"]/text()').extract()
    log.info("video time list length:{}".format(len(video_time_list)))
    for item in video_time_list:
        log.info(item)
    return video_time_list


# 获取每个视频的url 和名称对应关系
def get_video_url_list(html, title_name_list):
    video_list = html.xpath('//div[contains(@class,"session-banner ")]/div[@class="img"]/@style').extract()
    url_list = []
    name_dic = {}
    log.info("video url list length:{}".format(len(video_list)))
    for index, item in enumerate(video_list):
        # 从 style 中获取视频的url
        # url = item[23:-13]
        url = item[item.index('http'):item.index('.mp4') + len('.mp4')]
        log.info("原始链接:-> {} -> {}".format(index,url))

        url = url.replace('thumbnail', 'transcoding')
        url_list.append(url)
        name = get_file_name(url)
        name_dic[name] = title_name_list[index]
        log.info("目的链接:-> {} -> {}".format(index, url))
    return url_list, name_dic


# 下载视频第二个版本
def download_video_from_html(filename):
    content = read_html(filename)
    html = Selector(text=content)

    title_list = get_video_name_list(html)
    # time_list = get_video_time_list(html)
    url_list, name_dic = get_video_url_list(html, title_list)

    # log.info(url_list)
    # log.info(name_dic)
    download_file_and_rename(url_list, name_dic,file_path="/home/sl/workspace/python/mafengwo/imagehome/video3")


if __name__ == '__main__':
    # download_video_from_json()
    download_video_from_html(file_index_html)
