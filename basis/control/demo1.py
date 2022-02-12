#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : demo1.py
# @Author: sl
# @Date  : 2022/1/22 - 下午11:46

import control
import numpy as np
import matplotlib.pyplot as plt


def sys1(KA):
    G = control.tf([5 * KA], [1, 34.5])
    P = control.feedback(G, control.tf([1], [1]))
    d = np.linspace(0, 3, 500)
    t, rec = control.step_response(P, d)
    plt.plot(t, rec)


def sys2():
    Kt = 0.216
    d = np.linspace(0, 20, 1000)
    G1 = control.tf([10], [1, 1, 0])
    FB1 = control.tf([Kt, 0], [1])
    G0 = control.feedback(G1, FB1)
    P = control.feedback(G0, control.tf([1], [1]))
    t, rec = control.step_response(P, d)
    plt.plot(t, rec)


if __name__ == "__main__":
    sys1(1)  # 一阶系统
    sys2()#二阶系统
    plt.show()
