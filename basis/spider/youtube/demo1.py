#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : demo1.py
# @Author: sl
# @Date  : 2022/2/12 - 下午1:55

import youtube_dl
import os
import webbrowser


def down_video(url_list):
    # // 保存有youtube链接的文件
    # with open("F:/work/youtube/url.txt", 'r', encoding="utf8") as f:
    #     quanbuURLS = f.readlines()
    # print(len(quanbuURLS))
    # count = 1

    # url_list = []
    total = len(url_list)

    for index, url in enumerate(url_list):
        print('开始下载第{}个'.format(index))
        os.chdir(r"/home/sl/workspace/data/video/youtube/audio")
        # os.system("youtube-dl --write-auto-sub \
        # --sub-lang es --write-auto-sub  -f m4a " + url)

        # 下载音频
        os.system("youtube-dl -f m4a " + url)
        # 下载中文字幕
        os.system("youtube-dl --write-sub --sub-lang zh-CN --skip-download " + url)
        os.system("youtube-dl --write-sub --sub-lang zh-Hans --skip-download " + url)
        os.system("youtube-dl --write-sub --sub-lang zh-Hant --skip-download " + url)
        os.system("youtube-dl --write-sub --sub-lang zh --skip-download " + url)

        print('第{}个下载完成,已完成{:.3f}'.format(index, index / total))


if __name__ == '__main__':
    pass
    url_list = [
        "https://youtu.be/MdUkC7Vz3rg",
    ]
    down_video(url_list)
