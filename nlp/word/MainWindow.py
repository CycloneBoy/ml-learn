#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : MainWindow.py
# @Author: sl
# @Date  : 2021/1/10 -  下午1:51
import sys, time
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt, QSize

import sys,os
import time

import xlwt
import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QTableWidgetItem
from PyQt5.QtCore import QTimer, QThread, pyqtSignal


from AppMainGuiDemo import Ui_MainWindow



class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)

        self.openfile_name = None
        self.path_openfile_name = None

        self.input_table = None
        self.input_table_header = None
        self.input_table_rows = None
        self.input_table_colunms = None
        self.data = None
        self.remove_data = None
        self.save_data = None
        self.init()




    def init(self):
        self._set_connect()



    # 设置按钮的信号与槽函数
    def _set_connect(self):
        '''
        设置程序逻辑
        '''
        self.bt_open.clicked.connect(self.open_excel)
        self.bt_process.clicked.connect(self.process_data)
        self.bt_save.clicked.connect( self.save_data_to_excel)

    # 打开文件
    def open_excel(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', 'Excel files(*.xlsx , *.xls)')
        self.path_openfile_name = openfile_name[0]
        self.show_table()

    # 显示表格
    def show_table(self):
        if len(self.path_openfile_name) > 0:
            self.input_table = pd.read_excel(self.path_openfile_name)
            print("文件:{}".format(self.path_openfile_name))
            self.input_table_rows = self.input_table.shape[0]
            self.input_table_colunms = self.input_table.shape[1]
            print("文件行数和列数:{} - {}".format(self.input_table_rows,self.input_table_colunms))
            self.input_table_header = self.input_table.columns.values.tolist()
            print("文件header:{}".format(self.input_table_header))
            print()

            self.tableWidget_bag_word.setColumnCount(self.input_table_colunms)
            self.tableWidget_bag_word.setRowCount(self.input_table_rows)
            self.tableWidget_bag_word.setHorizontalHeaderLabels(self.input_table_header)

            data = []
            list = []
            for j in range(self.input_table_colunms):
                one_column = []
                list.append(one_column)

            ###================遍历表格每个元素，同时添加到tablewidget中========================
            for i in range(self.input_table_rows):
                input_table_rows_values = self.input_table.iloc[[i]]
                # print(input_table_rows_values)
                input_table_rows_values_array = np.array(input_table_rows_values)
                input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
                # print(input_table_rows_values_list)
                one_row = []
                for j in range(self.input_table_colunms):
                    input_table_items_list = input_table_rows_values_list[j]
                    # print(input_table_items_list)
                    # print(type(input_table_items_list))

                    input_table_items = str(input_table_items_list)
                    newItem = QTableWidgetItem(input_table_items)
                    newItem.setTextAlignment(Qt.AlignHCenter|Qt.AlignVCenter)

                    self.tableWidget_bag_word.setItem(i, j, newItem)
                    # self.tableWidget_bag_word.setColumnWidth(i, 100)
                    one_row.append(input_table_items)
                    list[j].append(input_table_items)
                data.append(one_row)

            self.data = list
            self.save_data = list
            # print(self.data)
            # print(self.data)

            # 显示
        else:
            self.centralWidget.show()

    def show_table_list(self,show_data,header):
        colunms = len(show_data[0])
        rows = len(show_data)
        self.tableWidget_bag_word.setColumnCount(colunms)
        self.tableWidget_bag_word.setRowCount(rows)
        # self.tableWidget_bag_word.setHorizontalHeaderLabels(header)
        for i in range(rows):
            for j in range(colunms):
                input_table_items = str(show_data[i][j])
                newItem = QTableWidgetItem(input_table_items)
                newItem.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.tableWidget_bag_word.setItem(i, j, newItem)

    def process_data(self):
        print("去重前大小:{}".format(len(self.data[0])))
        res_data = []
        for column in self.data:
            res_list = list(set(column))

            # 排序
            # res_list.sort()
            res_list.sort(key=lambda x:len(x),reverse=True)
            res_list = self.remove_same_prefix(res_list)

            res_data.append(res_list)

        # res_data_row = self.show_row_column(res_data)

        print("去重后大小:{}".format(len(res_data[0])))
        remove_input_str = self.read_input()
        res_data_2 = []
        for data in res_data:
            res_data2 = self.remove_input(data, remove_input_str)
            res_data_2.append(res_data2)

        res_data_row = np.transpose(res_data_2)
        # self.data = res_data_row

        self.save_data = res_data_row
        self.show_table_list(res_data_row,"None")

    def show_row_column(self,data):
        res_data = np.array([data]).T
        print(res_data)
        return res_data

    # 出去 重复的前缀的字符串
    def remove_same_prefix(self,data):
        res_list = list(set(data))
        res_list.sort(key=lambda x: len(x), reverse=True)

        res = []
        before = ""
        for index, word in enumerate(res_list):
            if index == 0:
                before = word
                res.append(before)
            else:
                flag = False
                for one_word in res:
                    if one_word.startswith(word):
                        flag = True
                        break
                if not flag:
                    res.append(word)

        # remove_input_str = self.read_input()
        # res = self.remove_input(res, remove_input_str)

        res.sort()
        return res

    def read_input(self):
        self.remove_data =self.plainTextEdit.toPlainText().split(os.linesep)
        print("需要移除的词组:{}".format(self.remove_data))
        return  self.remove_data

    def remove_input(self,data,remove_data):
        remove_set = set(remove_data)
        res_list = list(data)
        res = []
        for word in res_list:
            if word not in remove_set:
                    res.append(word)
        return  res

    def save_data_to_excel(self):
        show_data = self.save_data
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        openfile_excel,_ = QFileDialog.getSaveFileName(self, '选择保存位置', './', 'Excel files(*.xlsx , *.xls)')
        sheet = book.add_sheet('结果', cell_overwrite_ok=True)
        colunms = len(show_data[0])
        rows = len(show_data)
        for i in range(rows):
            for j in range(colunms):
                input_table_items = str(show_data[i][j])
                sheet.write(i, j, input_table_items)
        print("保存结果:{}".format(openfile_excel))
        book.save(openfile_excel)
        pass



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())