#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : TextEdit.py
# @Author: sl
# @Date  : 2020/9/25 - 下午10:53


#conding=utf-8

from PyQt5.QtWidgets import QWidget, QApplication, QColorDialog, QFontDialog, QTextEdit, QFileDialog
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDialog, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QIcon, QTextDocument
from PyQt5.QtCore import QDir, QFile
import sys


class NotePad(QMainWindow):

    fileName = "./newFile.txt"

    def __init__(self):
        super(NotePad, self).__init__()
        self.setGeometry(75, 75, 250, 430)
        self.setWindowTitle("myNotePad")
        self.setWindowIcon(QIcon("./image/gif/536717.gif"))

        self.textEdit = QTextEdit(self)
        self.setCentralWidget(self.textEdit)

        self.createMenu()
        self.show()

    def createMenu(self): #创建菜单
        self.fileMenu()
        self.editMenu()
        self.formatMenu()
        self.helpMenu()
        pass

    def fileMenu(self): #创建文件菜单
        menu = self.menuBar().addMenu("文件(&F)")

        newfile         = menu.addAction("新建(&N)", self.newFile_triggered)
        openfile        = menu.addAction("打开(&O)", self.openFile_triggered)
        savefile        = menu.addAction("保存(&S)", self.saveFile_triggered)
        savefileAs      = menu.addAction("另存为(&A)...", self.savaFileAs_triggered)
        exitfile        = menu.addAction("退出(&X)", self.exit_triggered)

        newfile. setShortcut("Ctrl+N")
        openfile.setShortcut("Ctrl+O")
        savefile.setShortcut("Ctrl+S")

    def editMenu(self): #创建编辑菜单
        menu = self.menuBar().addMenu("编辑(&E)")

        revokeEdit   = menu.addAction("撤销(&U)", self.revoke_triggered)
        recoveryEdit = menu.addAction("恢复(&R)", self.recovery_triggered)
        cutEdit      = menu.addAction("剪切(&T)", self.cut_triggered)
        copyEdit     = menu.addAction("复制(&C)", self.copy_triggered)
        pasteEdit    = menu.addAction("粘贴(&P)", self.paste_triggered)
        delEdit      = menu.addAction("删除(&L)", self.del__triggered)
        findEdit     = menu.addAction("查找(&F)", self.find_triggered)
        findNextEdit = menu.addAction("查找下一个(&N)")
        replaceEdit  = menu.addAction("替换(&E)...")
        gotoEdit     = menu.addAction("转到(&D)...")
        checkAllEdit = menu.addAction("全选(&A)")

        revokeEdit.setShortcut("Ctrl+Z") #设置快捷键
        recoveryEdit.setShortcut("Ctrl+Shift+Z")
        cutEdit.setShortcut("Ctrl+X")
        copyEdit.setShortcut("Ctrl+C")
        pasteEdit.setShortcut("Ctrl+V")
        delEdit.setShortcut("Del")
        findEdit.setShortcut("Ctrl+F")
        findNextEdit.setShortcut("F3")
        replaceEdit.setShortcut("Ctrl+H")
        gotoEdit.setShortcut("Ctrl+G")
        checkAllEdit.setShortcut("Ctrl+A")

        #设置为初始不可选区状态
        # revokeEdit.setEnabled(False)
        # cutEdit.setEnabled(False)
        # copyEdit.setEnabled(False)
        # delEdit.setEnabled(False)
        # findEdit.setEnabled(False)
        # findNextEdit.setEnabled(False)

    def formatMenu(self):  # 创建格式菜单
        self.menuBar().addMenu("Format(&O)")
        pass

    def helpMenu(self):  # 创建帮助菜单
        self.menuBar().addMenu("Help(&H)")
        pass


    def newFile_triggered(self): #新建文件
        self.textEdit.setText('')
        self.fileName = "./newFile.txt"

    def openFile_triggered(self): #打开文件
        filename, fileType = QFileDialog.getOpenFileName(self, "打开文件", './', "Image Files(*.jpg *.png *.txt *.py)")
        if len(fileType) != 0:
            self.fileName = filename
            text = ' '
            with open(self.fileName, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    text += line

            self.textEdit.setPlainText(text)

    def saveFile_triggered(self): #保存文件
        with open(self.fileName, 'w', encoding='utf-8') as f:
            text = self.textEdit.toPlainText()
            f.writelines(text)

    def savaFileAs_triggered(self): #文件另存为
        filename, fileType = QFileDialog.getSaveFileName(self, '保存文件')
        if len(fileType) != 0:
            with open(filename, 'w', encoding='utf-8') as f:
                text = self.textEdit.toPlainText()
                f.writelines(text)

    def exit_triggered(self):     #退出
        self.close()

    def revoke_triggered(self):   #撤销
        self.textEdit.undo()

    def recovery_triggered(self): #恢复
        self.textEdit.redo()

    def cut_triggered(self):
        self.textEdit.cut()

    def copy_triggered(self):
        self.textEdit.copy()

    def paste_triggered(self):
        self.textEdit.paste()

    def del__triggered(self):
        self.textEdit.deleteLater()

    def find_triggered(self):
        findDlg = QDialog(self)
        findDlg.setWindowTitle('查找...')

        self.find_textLineEdit = QLineEdit(findDlg)
        find_next_button = QPushButton('查找下一个', findDlg)
        find_last_button = QPushButton('查找上一个', findDlg)

        v_layout = QVBoxLayout(self)
        v_layout.addWidget(find_last_button)
        v_layout.addWidget(find_next_button)

        h_layout = QHBoxLayout(findDlg)
        h_layout.addWidget(self.find_textLineEdit)
        h_layout.addLayout(v_layout)

        find_last_button.clicked.connect(self.show_findLast)
        find_next_button.clicked.connect(self.show_findNext)


        findDlg.show()

    def show_findLast(self):
        find_text = self.find_textLineEdit.text()
        print(find_text)
        print(self.textEdit.find(find_text, QTextDocument.FindBackward))
        # if self.textEdit.find(find_text, QTextDocument.find):
        #     QMessageBox.warning(self, '查找', '找不到 {}'.format(find_text))

    def show_findNext(self):
        find_text = self.find_textLineEdit.text()
        print(find_text)
        print(self.textEdit.find(find_text, QTextDocument.FindBackward))
        if self.textEdit.find(find_text, QTextDocument.FindBackward):
            QMessageBox.warning(self, '查找', '找不到 {}'.format(find_text))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = NotePad()
    sys.exit(app.exec_())