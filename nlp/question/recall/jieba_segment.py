#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : jieba_segment.py
# @Author: sl
# @Date  : 2021/5/30 -  下午10:23

"""
结巴分词器
"""

import jieba
import codecs

from nlp.question.data.model import QaModel
from util.constant import TEST_QA_1
from util.nlp_utils import stop_words


class Seg(object):

    def __init__(self):
        self.stopwords = set(stop_words())

    def load_user_dict(self, filename):
        jieba.load_userdict(filename)

    def cut(self, sentence, stopword=True, for_search=False, cut_all=False):
        if for_search:
            seg_list = jieba.cut_for_search(sentence)
        else:
            seg_list = jieba.cut(sentence, cut_all)
        results = []
        for seg in seg_list:
            if stopword and seg in self.stopwords:
                continue
            results.append(seg)

        return results


if __name__ == '__main__':

    qa = QaModel(TEST_QA_1)
    print(qa)

    seg = Seg()
    cut1 = seg.cut(qa.question)
    print(cut1)

    cut2 = seg.cut(qa.answer)
    print(cut2)

    cut1 = seg.cut(qa.question,for_search=True)
    print(cut1)

    cut2 = seg.cut(qa.answer,for_search=True)
    print(cut2)
    pass

