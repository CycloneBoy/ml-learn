#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: sl
# @Date  : 2020/11/25 - 下午10:18

import torch

import paddlehub

import paddlehub as hub
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_torch():
    print(torch.__version__)  # 查看torch当前版本号

    print(torch.version.cuda)  # 编译当前版本的torch使用的cuda版本号

    print(torch.cuda.is_available())  # 查看当前cuda是否可用于当前版本的Torch，如果输出True，则表示可用

def test_face():
    face_landmark = hub.Module(name="face_landmark_localization")

    # Replace face detection module to speed up predictions but reduce performance
    # face_landmark.set_face_detector_module(hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320"))

    result = face_landmark.keypoint_detection(images=[cv2.imread('/home/sl/data/test_image.jpg')],
                                              use_gpu=True,output_dir='/home/sl/data/face_landmark_out',
                                              visualization=True                                              )
    # or
    # result = face_landmark.keypoint_detection(paths=['/PATH/TO/IMAGE'])
    print(result)

def test_face2():
    src_img = cv2.imread('//test_sample.jpg')

    module = hub.Module(name="face_landmark_localization")
    result = module.keypoint_detection(images=[src_img])

    tmp_img = src_img.copy()
    for index, point in enumerate(result[0]['data'][0]):
        # cv2.putText(img, str(index), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), -1)
        cv2.circle(tmp_img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

    res_img_path = 'face_landmark.jpg'
    cv2.imwrite(res_img_path, tmp_img)

    img = mpimg.imread(res_img_path)
    # 展示预测68个关键点结果
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # paddlehub.server_check()
    test_face()
