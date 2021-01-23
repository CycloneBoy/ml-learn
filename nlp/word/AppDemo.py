#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : AppDemo.py
# @Author: sl
# @Date  : 2021/1/10 -  下午1:50


import matplotlib

from MainWindow import MainWindow

matplotlib.use("Qt5Agg")  # 声明使用QT5

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

if __name__ == '__main__':
    # application 对象
    app = QApplication(sys.argv)

    # QMainWindow对象
    mainwindow = MainWindow()

    # 这是qt designer实现的Ui_MainWindow类
    # ui_components = Ui_MainWindow(mainwindow)
    # 调用setupUi()方法，注册到QMainWindow对象
    # ui_components.setupUi(mainwindow)


    # 显示
    mainwindow.show()

    sys.exit(app.exec_())