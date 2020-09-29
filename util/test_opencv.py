#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_opencv.py
# @Author: sl
# @Date  : 2020/9/19 - 下午10:37

import cv2 as cv

print(cv.__version__)

img = cv.imread("/home/sl/workspace/python/a2020/ml-learn/data/image/test.jpg",1)
cv.imshow('Image',img)
cv.waitKey(0)
