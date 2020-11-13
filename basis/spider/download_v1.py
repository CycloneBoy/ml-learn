#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : download_v1.py
# @Author: sl
# @Date  : 2020/11/8 - 下午10:26

import json
import os
import urllib3
from basis.utils.multi_thread_download import multi_download, stop_treads,log,setImageDir

filename = "course/data2.json"
http= urllib3.PoolManager()

def read_data(filename):
    video_list = []
    # 读取数据
    with open(filename, 'r') as f:
        response = json.load(f)
        # print(response)

        list = response['data']['resource_list']
        print(list)

        for key in list.keys():
            video = {}
            value = list[key]
            # print("{} ->{}".format(key, value))

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

def download(url,file_name,file_path='/home/sl/workspace/data/test'):
    r=http.request('GET',url)
    save_path = os.path.join(file_path,file_name)
    with open(save_path, "wb") as code:
        code.write(r.data)

# 下载文件
def download_v2(url_list,file_path='/home/sl/workspace/data/test'):
    threads,starttime = multi_download(url_list,imageDir="video")
    stop_treads(threads,starttime,url_list)

def rename_file(file_path,name_dic):
    fileList = os.listdir(file_path)
    for name in fileList:
        old_name = os.path.join(file_path,name)
        print(old_name)
        if name in name_dic:
            new_name = os.path.join(file_path,name_dic[name])
            os.rename(old_name,new_name)
            log.info("文件重命名:{} 为:{} ".format(old_name,new_name))


def get_file_name(file_url):
    file = str(file_url).split('/')
    name = file[len(file) - 1]
    return name



if __name__ == '__main__':
    video_list =read_data(filename)

    url_list = []
    name_dic = {}
    for video in video_list:
        file_url = video['url']
        file_name = video['file_name']
        name = get_file_name(file_url)
        print("{} -> {}".format(file_name,file_url))
        url_list.append(file_url)
        name_dic[name] = file_name

    download_v2(url_list)
    file_path = os.path.join(setImageDir,'video')
    rename_file(file_path,name_dic)




