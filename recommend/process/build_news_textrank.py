#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : build_news_textrank.py
# @Author: sl
# @Date  : 2021/9/27 - 下午5:11


"""
处理新闻的离线文章画像
 TextRank 抽取关键词
"""

import os

import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs

# 分词
from recommend.utils.constants import TABLE_TEXT_RANK
from recommend.utils.spark_utils import get_spark, read_mysql_to_df, save_df_to_mysql
from util.nlp_utils import stop_words


def textrank(partition):

    abspath = "../properties"

    # 结巴加载用户词典
    userDict_path = os.path.join(abspath, "NewsKeywords.txt")
    jieba.load_userdict(userDict_path)

    # 停用词文本
    stopwords_path = os.path.join(abspath, "stopwords.txt")

    def get_stopwords_list():
        """返回stopwords列表"""
        stopwords_list = [i.strip()
                          for i in codecs.open(stopwords_path).readlines()]
        return stopwords_list

    # 所有的停用词列表
    stopwords_list = stop_words()

    class TextRank(jieba.analyse.TextRank):
        def __init__(self, window=20, word_min_len=2):
            super(TextRank, self).__init__()
            self.span = window  # 窗口大小
            self.word_min_len = word_min_len  # 单词的最小长度
            # 要保留的词性，根据jieba github ，具体参见https://github.com/baidu/lac
            self.pos_filt = frozenset(
                ('n', 'x', 'eng', 'f', 's', 't', 'nr', 'ns', 'nt', "nw", "nz", "PER", "LOC", "ORG"))

        def pairfilter(self, wp):
            """过滤条件，返回True或者False"""

            if wp.flag == "eng":
                if len(wp.word) <= 2:
                    return False

            if wp.flag in self.pos_filt and len(wp.word.strip()) >= self.word_min_len \
                    and wp.word.lower() not in stopwords_list:
                return True
    # TextRank过滤窗口大小为5，单词最小为2
    textrank_model = TextRank(window=5, word_min_len=2)
    allowPOS = ('n', "x", 'eng', 'nr', 'ns', 'nt', "nw", "nz", "c")

    for row in partition:
        news = f"{row.category},{row.title},{row.content},"
        tags = textrank_model.textrank(news, topK=20, withWeight=True, allowPOS=allowPOS, withFlag=False)
        for tag in tags:
            yield row.nid, row.category, tag[0], tag[1]



def calc_textrank(spark):
    article_dataframe = read_mysql_to_df(spark,table='t_news_analysis')
    # 计算textrank
    textrank_keywords_df = article_dataframe.rdd.mapPartitions(textrank).toDF(
        ["article_id", "channel_id", "keyword", "textrank"])

    textrank_keywords_df.show()
    # textrank_keywords_df.write.insertInto("textrank_keywords_values")
    save_df_to_mysql(textrank_keywords_df,table=TABLE_TEXT_RANK)


if __name__ == '__main__':
    spark = get_spark()

    calc_textrank(spark)