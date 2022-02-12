#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : demo4.py
# @Author: sl
# @Date  : 2022/1/23 - 下午8:28

import numpy as np
import matplotlib.pyplot as plt


# 适应度函数,求取最大值
# 因为GA函数是求最小值，所以我在适应度函数上加一个负号
# GA要求输入维度2维及其以上，所以我传入2个参数，第二维x2不用
def fitness(x):
    x1, x2 = x
    # x1=x
    return -(x1 + 16 * np.sin(5 * x1) + 10 * np.cos(4 * x1))


# 个体类
from sko.GA import GA

ga = GA(func=fitness, n_dim=2, size_pop=50, max_iter=800, lb=[-10, 0], ub=[10, 0], precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x[0], '\n', 'best_y:', -best_y)


def func(x):
    return x + 16 * np.sin(5 * x) + 10 * np.cos(4 * x)


x = np.linspace(-10, 10, 100000)
y = func(x)

plt.plot(x, y)
plt.scatter(best_x[0], -best_y, c='r', label='best point')

plt.legend()
plt.show()

if __name__ == '__main__':
    pass
