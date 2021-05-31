#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : sentence.py
# @Author: sl
# @Date  : 2021/5/31 -  下午10:21


class Sentence(object):

    def __init__(self, sentence, seg, id=0):
        self.score = 0
        self.id = id
        self.origin_sentence = sentence
        self.cut_sentence = self.cut(seg)

    def cut(self, seg):
        """句子分词"""
        return seg.cut(self.origin_sentence, for_search=True)

    def get_cut_sentence(self):
        """获取切词后的词列表"""
        return self.cut_sentence

    def get_origin_sentence(self):
        """获取原句子"""
        return self.origin_sentence

    def set_score(self, score):
        """设置该句子得分"""
        self.score = score
