#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project ml-learn
# @File  : download_demo.py
# @Author: sl
# @Date  : 2022/5/19 - 下午11:29
import urllib.request

import requests
import subprocess
import os

from util.logger_utils import logger

"""
curl 'https://encrypt-k-vod.xet.tech/9764a7a5vodtransgzp1252524126/c14b52f45285890816971350990/drm/v.f421220.ts?start=136719184&end=137218527&type=mpegts&exper=0&sign=6519a132e1a730191652fe31fa3b783f&t=6287b487&us=tUOMbIpmm40e' \
  -H 'Connection: keep-alive' \
  -H 'sec-ch-ua: " Not;A Brand";v="99", "Google Chrome";v="97", "Chromium";v="97"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36' \
  -H 'sec-ch-ua-platform: "Linux"' \
  -H 'Accept: */*' \
  -H 'Origin: https://appixgujdjj6027.h5.xiaoeknow.com' \
  -H 'Sec-Fetch-Site: cross-site' \
  -H 'Sec-Fetch-Mode: cors' \
  -H 'Sec-Fetch-Dest: empty' \
  -H 'Referer: https://appixgujdjj6027.h5.xiaoeknow.com/' \
  -H 'Accept-Language: en-US,en;q=0.9' \
  --compressed
"""


if __name__ == '__main__':
    urlsource = 'https://encrypt-k-vod.xet.tech/9764a7a5vodtransgzp1252524126/c14b52f45285890816971350990/drm/v.f421220.ts'
    folderName = './football'
    folderPathName = folderName

    try:
        os.makedirs(folderPathName)
    except:
        pass

    url_list = [i for i in range(0, 2490111, 494255)]
    for index,u in enumerate(range(0,len(url_list) - 1)):
        start = url_list[index]
        end = url_list[index+1]

        params = f"?start={start}&end={end}&type=mpegts&exper=0&sign=6519a132e1a730191652fe31fa3b783f&t=6287b487&us=tUOMbIpmm40e"

        fileName = f"{start}_{end}.ts"
        filePathName = folderPathName + fileName
        url = urlsource + params
        logger.info(f"index: {u} - {url}")

        urllib.request.urlretrieve(url, filePathName)
        print("{} done".format(fileName))

    print("完成")
