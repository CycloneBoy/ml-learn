#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : TextRank4Sentence.py
# @Author: sl
# @Date  : 2021/7/3 -  下午11:00

from . import util

class TextRank4Sentence(object):
    def __init__(self, stop_words_file=None,
                 allow_speech_tags=util.allow_speech_tags,
                 delimiters=util.sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file  --  str，停止词文件路径，若不是str则是使用默认停止词文件
        delimiters       --  默认值是`?!;？！。；…\n`，用来将文本拆分为句子。

        Object Var:
        self.sentences               --  由句子组成的列表。
        self.words_no_filter         --  对sentences中每个句子分词而得到的两级列表。
        self.words_no_stop_words     --  去掉words_no_filter中的停止词而得到的两级列表。
        self.words_all_filters       --  保留words_no_stop_words中指定词性的单词而得到的两级列表。
        """