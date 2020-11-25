#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_read.py
# @Author: sl
# @Date  : 2020/11/20 - 下午9:28
import os

import cv2
import numpy as np

from deep.opencv.LicensePlateRecognition.utils.Constants import OPENCV_DIR

image_dir = os.path.join(OPENCV_DIR, "car")

debug_mode = True

def debug(name,img,debug_mode=True):
    """
    显示调试图像
    :param name:
    :param img:
    :param debug_mode:
    :return:
    """
    if debug_mode:
        cv2.imshow(name, img)

def img_read_photo(filename, flags=cv2.IMREAD_COLOR):
    """
    该函数能够读取磁盘中的图片文件，默认以彩色图像的方式进行读取
    :param filename: 指的图像文件名（可以包括路径）
    :param flags: 用来表示按照什么方式读取图片，有以下选择（默认采用彩色图像的方式）
              IMREAD_COLOR 彩色图像
              IMREAD_GRAYSCALE 灰度图像
              IMREAD_ANYCOLOR 任意图像

    :return: 返回图片的通道矩阵
    """
    return cv2.imread(filename, flags)


def resize_photo(img_mat, MAX_WIDTH=1000):
    """
    这个函数的作用就是来调整图像的尺寸大小，当输入图像尺寸的宽度大于阈值（默认1000），我们会将图像按比例缩小

    拓展：OpenCV自带的cv2.resize()函数可以实现放大与缩小，函数声明如下：
            cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) → dst
        其参数解释如下：
            src 输入图像矩阵
            dsize 二元元祖（宽，高），即输出图像的大小
            dst 输出图像矩阵
            fx 在水平方向上缩放比例，默认值为0
            fy 在垂直方向上缩放比例，默认值为0
            interpolation 插值法，如INTER_NEAREST，INTER_LINEAR，INTER_AREA，INTER_CUBIC，INTER_LANCZOS4等


    :param img_mat:  是输入的图像数字矩阵
    :param MAX_WIDTH:
    :return:  经过调整后的图像数字矩阵
    """
    img = img_mat
    rows, cols = img.shape[:2]
    if cols > MAX_WIDTH:
        change_rate = MAX_WIDTH / cols
        img = cv2.resize(img, (MAX_WIDTH, int(rows * change_rate)), interpolation=cv2.INTER_AREA)
    return img

def predict(img_mat):
    """
    # # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中

    :param img_mat: 原始图像的数字矩阵
    :return: gray_img_原始图像经过高斯平滑后的二值图
          contours是找到的多个轮廓
    """
    img_copy = img_mat.copy()
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    debug('gray_img', gray_img)
    gray_img_ = cv2.GaussianBlur(gray_img,(5,5),0,0,cv2.BORDER_DEFAULT)
    debug('gray_img_', gray_img_)

    kernel = np.ones((23,23),np.uint(8))
    img_opening = cv2.morphologyEx(gray_img,cv2.MORPH_OPEN,kernel)
    debug('img_opening', img_opening)
    img_opening2 = cv2.addWeighted(gray_img,1,img_opening,-1,0)
    debug('img_opening2', img_opening2)
    # 找到图像边缘
    ret,img_thresh = cv2.threshold(img_opening2,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    debug('img_thresh', img_thresh)
    img_edge = cv2.Canny(img_thresh,100,200)
    debug('img_edge', img_edge)

    # # 使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((10,10),np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge,cv2.MORPH_CLOSE,kernel)
    debug('img_edge1_open', img_edge1)
    img_edge2 = cv2.morphologyEx(img_edge1,cv2.MORPH_OPEN,kernel)
    debug('img_edge1_close', img_edge2)
    # # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    contours, hierarchy = cv2.findContours(img_edge2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return  gray_img_,contours

def chose_licence_plate(contours,min_area = 2000):
    """
    这个函数根据车牌的一些物理特征（面积等）对所得的矩形进行过滤
    输入：contours是一个包含多个轮廓的列表，其中列表中的每一个元素是一个N*1*2的三维数组
    输出：返回经过过滤后的轮廓集合

    拓展：
    （1） OpenCV自带的cv2.contourArea()函数可以实现计算点集（轮廓）所围区域的面积，函数声明如下：
            contourArea(contour[, oriented]) -> retval
        其中参数解释如下：
            contour代表输入点集，此点集形式是一个n*2的二维ndarray或者n*1*2的三维ndarray
            retval 表示点集（轮廓）所围区域的面积
    （2） OpenCV自带的cv2.minAreaRect()函数可以计算出点集的最小外包旋转矩形，函数声明如下：
             minAreaRect(points) -> retval
        其中参数解释如下：
            points表示输入的点集，如果使用的是Opencv 2.X,则输入点集有两种形式：一是N*2的二维ndarray，其数据类型只能为 int32
                                    或者float32， 即每一行代表一个点；二是N*1*2的三维ndarray，其数据类型只能为int32或者float32
            retval是一个由三个元素组成的元组，依次代表旋转矩形的中心点坐标、尺寸和旋转角度（根据中心坐标、尺寸和旋转角度
                                    可以确定一个旋转矩形）
    （3） OpenCV自带的cv2.boxPoints()函数可以根据旋转矩形的中心的坐标、尺寸和旋转角度，计算出旋转矩形的四个顶点，函数声明如下：
             boxPoints(box[, points]) -> points
        其中参数解释如下：
            box是旋转矩形的三个属性值，通常用一个元组表示，如（（3.0，5.0），（8.0，4.0），-60）
            points是返回的四个顶点，所返回的四个顶点是4行2列、数据类型为float32的ndarray，每一行代表一个顶点坐标

    :param contours: contours是一个包含多个轮廓的列表，其中列表中的每一个元素是一个N*1*2的三维数组
    :param min_area: 最小面积
    :return: 返回经过过滤后的轮廓集合
    """
    temp_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            temp_contours.append(contour)

    car_plates = []
    for temp_contour in temp_contours:
        rect_tupple =cv2.minAreaRect(temp_contour)
        rect_width ,rect_height = rect_tupple[1]
        if rect_width < rect_height:
            rect_width,rect_height = rect_height, rect_width
        aspect_ratio = rect_width/rect_height
        # 车牌正常情况下宽高比在2 - 5.5之间
        if aspect_ratio > 2 and aspect_ratio < 5.5:
            car_plates.append(temp_contour)
            rect_vertices = cv2.boxPoints(rect_tupple)
            rect_vertices = np.int0(rect_vertices)

    return car_plates


def license_segment(car_plates):
    """
        此函数根据得到的车牌定位，将车牌从原始图像中截取出来，并存在当前目录中。
    :param car_plates:  是经过初步筛选之后的车牌轮廓的点集
    :return: 车牌的存储名字
    """
    if len(car_plates) == 1:
        for car_plate in car_plates:
            row_min,col_min = np.min(car_plate[:,0,:],axis=0)
            row_max,col_max = np.max(car_plate[:,0,:],axis=0)
            cv2.rectangle(img,(row_min,col_min),(row_max,col_max),(0,255,0),2)
            card_img = img[col_min:col_max,row_min:row_max]
            debug("license_segment-img",img)

        name =  "card_img.png"
        card_img_filename = os.path.join(image_dir,name)
        cv2.imwrite( card_img_filename,card_img)
        debug("card_img", card_img)
    return card_img_filename


# 根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def remove_plate_upanddown_border(card_img):
    """
    这个函数将截取到的车牌照片转化为灰度图，然后去除车牌的上下无用的边缘部分，确定上下边框
    输入： card_img是从原始图片中分割出的车牌照片
    输出: 在高度上缩小后的字符二值图片
    """
    plate_Arr = cv2.imread(card_img)
    # plate_gray_Arr = cv2.cvtColor(plate_Arr, cv2.COLOR_BGR2GRAY)
    plate_gray_Arr = cv2.cvtColor(plate_Arr, cv2.COLOR_BGR2GRAY)
    ret, plate_binary_img = cv2.threshold(plate_gray_Arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    row_histogram = np.sum(plate_binary_img, axis=1)  # 数组的每一行求和
    row_min = np.min(row_histogram)
    row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
    row_threshold = (row_min + row_average) / 2
    wave_peaks = find_waves(row_threshold, row_histogram)
    # 接下来挑选跨度最大的波峰
    wave_span = 0.0
    for wave_peak in wave_peaks:
        span = wave_peak[1] - wave_peak[0]
        if span > wave_span:
            wave_span = span
            selected_wave = wave_peak
    plate_binary_img = plate_binary_img[selected_wave[0]:selected_wave[1], :]
    debug("plate_binary_img", plate_binary_img)

    return plate_binary_img

    ##################################################
    # 测试用
    # print( row_histogram )
    # fig = plt.figure()
    # plt.hist( row_histogram )
    # plt.show()
    # 其中row_histogram是一个列表，列表当中的每一个元素是车牌二值图像每一行的灰度值之和，列表的长度等于二值图像的高度
    # 认为在高度方向，跨度最大的波峰为车牌区域
    cv2.imshow("plate_gray_Arr", plate_binary_img[selected_wave[0]:selected_wave[1], :])
    ##################################################


if __name__ == '__main__':
    image_filename = os.path.join(image_dir, "car1.png")
    img = img_read_photo(image_filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img', img)
    # cv2.imshow('gray_img', gray_img)

    gray_img_,contours = predict(img)
    print(contours)

    car_plates = chose_licence_plate(contours)
    name =license_segment(car_plates)

    remove_plate_upanddown_border(name)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
