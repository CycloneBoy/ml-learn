#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : AppMainGui.py
# @Author: sl
# @Date  : 2021/1/9 -  上午8:46

import sys,os
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtCore import QTimer, QThread, pyqtSignal




class Ui_MainWindow(object):
    def __init__(self):
        pass

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 879)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.sa = QtWidgets.QScrollArea(self.centralWidget)
        self.sa.setWidgetResizable(True)
        self.sa.setObjectName("sa")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1004, 859))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollAreaWidgetContents.sizePolicy().hasHeightForWidth())
        self.scrollAreaWidgetContents.setSizePolicy(sizePolicy)
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.tabWidget = QtWidgets.QTabWidget(self.scrollAreaWidgetContents)
        self.tabWidget.setGeometry(QtCore.QRect(10, 170, 1001, 691))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_bag_word = QtWidgets.QWidget()
        self.tab_bag_word.setObjectName("tab_bag_word")
        self.scrollArea = QtWidgets.QScrollArea(self.tab_bag_word)
        self.scrollArea.setGeometry(QtCore.QRect(0, 0, 991, 661))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 989, 659))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.tableWidget_bag_word = QtWidgets.QTableWidget(self.scrollAreaWidgetContents_2)
        self.tableWidget_bag_word.setGeometry(QtCore.QRect(0, 0, 981, 661))
        self.tableWidget_bag_word.setObjectName("tableWidget_bag_word")
        self.tableWidget_bag_word.setColumnCount(0)
        self.tableWidget_bag_word.setRowCount(0)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents_2)
        self.tabWidget.addTab(self.tab_bag_word, "")
        self.tab_process_word = QtWidgets.QWidget()
        self.tab_process_word.setObjectName("tab_process_word")
        self.tabWidget.addTab(self.tab_process_word, "")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.scrollAreaWidgetContents)
        self.plainTextEdit.setGeometry(QtCore.QRect(140, 30, 531, 70))
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label.setGeometry(QtCore.QRect(20, 50, 101, 17))
        self.label.setObjectName("label")
        self.bt_process = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.bt_process.setGeometry(QtCore.QRect(690, 70, 89, 25))
        self.bt_process.setObjectName("bt_process")
        self.bt_open = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.bt_open.setGeometry(QtCore.QRect(690, 30, 89, 25))
        self.bt_open.setObjectName("bt_open")
        self.bt_save = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.bt_save.setGeometry(QtCore.QRect(800, 30, 89, 25))
        self.bt_save.setObjectName("bt_save")
        self.sa.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout.addWidget(self.sa)
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_bag_word), _translate("MainWindow", "词包"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_process_word), _translate("MainWindow", "取词"))
        self.label.setText(_translate("MainWindow", "请输入关键词:"))
        self.bt_process.setText(_translate("MainWindow", "进行取词"))
        self.bt_open.setText(_translate("MainWindow", "打开文件"))
        self.bt_save.setText(_translate("MainWindow", "保存文件"))