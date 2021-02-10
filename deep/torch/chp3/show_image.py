#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : show_image.py
# @Author: sl
# @Date  : 2021/2/10 -  下午1:20


import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False


if __name__ == '__main__':
    # 添加一个标题
    plt.title(u'知乎')
    # 给y轴加标记
    plt.ylabel(u'知乎')
    plt.show()

    mpl.rcParams[u'font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.title(u'知乎')

