#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : download_v1.py
# @Author: sl
# @Date  : 2020/11/8 - 下午10:26

import json
import os
import urllib3

filename = "data.json"
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

if __name__ == '__main__':
    video_list =read_data(filename)

    for video in video_list:
        print(video['url'])
        # download(video['url'],video['file_name'])





