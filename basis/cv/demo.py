#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : demo.py
# @Author: sl
# @Date  : 2021/12/25 - 下午4:00
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

"""
有一个2进制文件(1M~2M大小)，可以用16进制查看，
里面包括一张图片(大小可能是1360*1024)和一些其他信息，
图片是可能LZW压缩也可能是raw其他方法压缩不确定，
需要解析出图片的代码.

1392640

5 - 1147542
6 - 1036549
7 - 1230648
"""

if __name__ == '__main__':
    img1 = "/home/sl/图片/test/309101-portrait.jpg"

    rows = 1360
    cols = 1024
    channels = 1

    # img1_mat = cv2.imread(img1)
    # cv2.imshow('demo',img1_mat)
    # cv2.waitKey()

    img_url1 = r'/home/sl/workspace/data/opencv/21KB000749.005.mmi'
    img_url2 = r'/home/sl/workspace/data/opencv/21KB000749.006.mmi'
    img_url3 = r'/home/sl/workspace/data/opencv/21KB000749.007.mmi'
    total_line = 0
    with open(img_url1, 'rb') as f:
        for line in f:
            total_line += 1
        a = f.read()

    print(f"line: {total_line}")
    a_temp = np.fromfile(img_url1, dtype=np.uint8)
    print(len(a_temp))

    total = 18 * 16
    head = a_temp[:total]
    print(head)

    for row in a_temp:
        print(f"{row}")

    # img = a_temp.reshape(rows, cols, channels)
    #
    # # 展示图像
    # cv2.imshow('Infared image-640*512-8bit', img)

    # img = cv2.imdecode(a_temp, cv2.IMREAD_COLOR)
    # # # 将bgr转为rbg
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # print(rgb_img)
    # # np.ndarray转IMAGE
    # a = Image.fromarray(rgb_img)
    # print(a)
    # # 显示图片
