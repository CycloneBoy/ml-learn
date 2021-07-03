#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : lead_3.py
# @Author: sl
# @Date  : 2021/7/3 -  下午12:07
from util.constant import TEST_SUMMARY_1, TEST_SUMMARY_DOC
from util.nlp_utils import cut_sentence

"""
Lead-3
一般来说，作者常常会在标题和文章开始就表明主题，
因此最简单的方法就是抽取文章中的前几句作为摘要。
常用的方法为 Lead-3，
即抽取文章的前三句作为文章的摘要。Lead-3 方法虽然简单直接，但却是非常有效的方法。
"""

class Lead3Sum(object):

    def __init__(self):
        self.algorithm = 'lead_3'

    def summarize(self,text,run_type='mix',num=3):
        """
            lead-s
        :param sentences: list
        :param type: str, you can choose 'begin', 'end' or 'mix'
        :return: list
        """

        sentences = cut_sentence(text)
        if len(sentences) < num:
            return  sentences

        num_min = min(num,len(sentences))
        if run_type =='begin':
            summers = sentences[0:num]
        elif run_type == 'end':
            summers = sentences[-num:]
        else:
            summers = [sentences[0]] + [sentences[-1]] + sentences[1:num - 1]
        summers_s = {}
        # 得分计算
        for i in range(len(summers)):
            if len(summers) - i == 1:
                summers_s[summers[i]] = (num - 0.75) / (num + 1)
            else:
                summers_s[summers[i]] = (num - i - 0.5) / (num + 1)
        score_sen = [(rc[1], rc[0]) for rc in sorted(summers_s.items(), key=lambda d: d[1], reverse=True)][0:num_min]
        return score_sen

if __name__ == '__main__':
    sen = TEST_SUMMARY_1[1]
    sen = TEST_SUMMARY_DOC

    l3 = Lead3Sum()
    for score_sen in l3.summarize(sen,run_type='mix',num=3):
        print(score_sen)