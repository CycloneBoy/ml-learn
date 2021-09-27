#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : DataModel.py
# @Author: sl
# @Date  : 2021/9/26 - 下午12:50

class ArticleModel:

    def __init__(self, id=None, title=None, content=None, category=None, news_time=None):
        self.id = id
        self.title = title
        self.content = content
        self.category = category
        self.news_time = news_time

    def __str__(self):
        return f"{self.category}|{self.id}|{self.title}|{self.content}|{self.news_time}"
