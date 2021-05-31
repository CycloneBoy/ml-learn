#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: sl
# @Date  : 2021/5/30 -  下午9:26


class QaModel(object):
    """问答模型"""

    def __init__(self, line=None, delimiter="|"):
        self.id = None
        self.question = None
        self.answer = None
        self.time = None
        self.delimiter = delimiter
        if line is not None:
            self.parse(line)

    def parse(self, row):
        line = row.split(self.delimiter)
        if len(line) >= 4:
            self.id = line[0]
            self.question = line[1]
            self.answer = line[2]
            self.time = line[3]

    def __str__(self):
        data = [self.id, self.question, self.answer, self.time]
        return self.delimiter.join(data)
