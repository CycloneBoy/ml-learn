#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : demo2.py
# @Author: sl
# @Date  : 2022/1/22 - 下午11:50

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

# 设置字体
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 定义数据
itor = 1
X1 = 5  # 横坐标固定为5，只改变小球的高度
X2 = 15  # 初始纵坐标
XP = 10  # 预设的纵坐标
v = 0  # 速度
a = 0  # 加速度

Kp = 0.2
Kd = 0.2
Err1 = X2 - XP
Err0 = 0
Max_itor = 50000
#
plt.figure(figsize=(12, 8), dpi=100)
plt.ion()

while itor < Max_itor:
    # 清空旧的画布
    plt.cla()
    plt.xlim((-1, 20))
    plt.ylim((-1, 20))
    plt.plot([-1, 20], [XP, XP], 'r-')

    # rgeion
    a = Kp * (1.0 * Err1 + Kd * (Err1 - Err0))  # 按PD算法计算要给予的加速度
    # a = Kp*(1.0*Err1)  # 不用PD算法
    v = v - a
    X2 = X2 + v
    # endregion

    Err0 = Err1  # 保留旧的误差
    Err1 = X2 - XP  # 计算新的误差

    plt.scatter(X1, X2,  # 散点的横、纵坐标序列
                s=30,  # 点
                c='blue',
                label="PD Algorithm")
    plt.legend()  # 显示图例说明
    itor = itor + 1
    plt.pause(0.008)

if __name__ == '__main__':
    pass
