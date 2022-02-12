#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : demo3.py
# @Author: sl
# @Date  : 2022/1/22 - 下午11:53

# pvtol-nested.py - inner/outer design for vectored thrust aircraft
# RMM, 5 Sep 09
#
# This file works through a fairly complicated control design and
# analysis, corresponding to the planar vertical takeoff and landing
# (PVTOL) aircraft in Astrom and Murray, Chapter 11.  It is intended
# to demonstrate the basic functionality of the python-control
# package.
#

from __future__ import print_function

import os
import matplotlib.pyplot as plt  # MATLAB plotting functions
from control.matlab import *  # MATLAB-like functions
import numpy as np

# System parameters
m = 4  # 飞行器质量
J = 0.0475  # 俯仰轴惯性
r = 0.25  # 力与质心距离（原文为：distance to center of force
g = 9.8  # 重力加速度
c = 0.05  # 阻尼系数（估计）

# 传递函数
Pi = tf([r], [J, 0, 0])  # inner loop (roll)
Po = tf([1], [m, c, 0])  # outer loop (position)

# Use state space versions
Pi = tf2ss(Pi)
Po = tf2ss(Po)

#
# 内控制环路设计
#
# This is the controller for the pitch dynamics.  Goal is to have
# fast response for the pitch dynamics so that we can use this as a
# control for the lateral dynamics
#

# Design a simple lead controller for the system
k, a, b = 200, 2, 50
Ci = k * tf([1, a], [1, b])  # lead compensator
Li = Pi * Ci

Si = feedback(1, Li)
Ti = Li * Si

Hi = parallel(feedback(Ci, Pi), -m * g * feedback(Ci * Pi, 1))

a, b, K = 0.02, 5, 2
Co = -K * tf([1, 0.3], [1, 10])  # another lead compensator
Lo = -m * g * Po * Co

# Finally compute the real outer-loop loop gain + responses
L = Co * Hi * Po
S = feedback(1, L)
T = feedback(L, 1)

plt.figure('时域响应')
y, t = step(T, T=np.linspace(0, 10, 100))
plt.plot(t, y)
plt.xlabel('time/s')
plt.ylabel('y(t)')
plt.grid()

# 对整个系统绘制伯德图
plt.figure('伯德图')
bode(L, np.logspace(-4, 3))

plt.figure('奈奎斯特图')

nyquist(L, (0.0001, 1000))
plt.axis([-4000, 300, -300000, 300000])

plt.figure('四个图')
gangof4(Hi * Po, Co)
plt.show()

if __name__ == '__main__':
    pass
