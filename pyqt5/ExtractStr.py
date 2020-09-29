#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : ExtractStr.py
# @Author: sl
# @Date  : 2020/9/24 - 下午9:57

from pyqt5.designer.MainWindow import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

if __name__ == '__main__':
    # application 对象
    app = QApplication(sys.argv)

    # QMainWindow对象
    mainwindow = QMainWindow()

    # 这是qt designer实现的Ui_MainWindow类
    ui_components = Ui_MainWindow()
    # 调用setupUi()方法，注册到QMainWindow对象
    ui_components.setupUi(mainwindow)

    # 显示
    mainwindow.show()

    sys.exit(app.exec_())