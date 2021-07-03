#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : textrank_textrank4zh.py
# @Author: sl
# @Date  : 2021/7/3 -  下午2:04

from textrank4zh import TextRank4Sentence

from util.constant import TEST_SUMMARY_1, TEST_SUMMARY_DOC


def textrank_zh(sentences,topK=6):
    tr4zh = TextRank4Sentence()
    tr4zh.analyze(text=sentences,lower=True,source='all_filters')
    res = tr4zh.get_key_sentences(num=topK)
    # res = sorted(res,key=lambda x:x['weight'],reverse=True)
    return res

if __name__ == '__main__':

    sen = TEST_SUMMARY_1[1]
    # sen = TEST_SUMMARY_DOC

    for score_sen in textrank_zh(sen,6):
        print(score_sen)