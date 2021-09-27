#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : build_news_tfidf.py
# @Author: sl
# @Date  : 2021/9/27 - 下午3:03


import os
import re

import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs

from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, IDF, IDFModel

from recommend.utils.constants import COUNT_VECTORIZER_MODEL_PATH, IDF_MODEL_PATH, WORDS_DF_PATH, MYSQL_URL, MYSQL_PROP, \
    TABLE_TFIDF
from recommend.utils.spark_utils import save_df_to_mysql, read_mysql_to_df, get_spark
from util.nlp_utils import stop_words

"""
处理新闻的离线文章画像
TFIDF TextRank 抽取关键词
"""


# 分词
def segmentation(partition):
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

    # 分词
    def cut_sentence(sentence):
        """对切割之后的词语进行过滤，去除停用词，保留名词，英文和自定义词库中的词，长度大于2的词"""
        # print(sentence,"*"*100)
        # eg:[pair('今天', 't'), pair('有', 'd'), pair('雾', 'n'), pair('霾', 'g')]
        seg_list = pseg.lcut(sentence)
        seg_list = [i for i in seg_list if i.flag not in stopwords_list]
        filtered_words_list = []
        for seg in seg_list:
            # print(seg)
            if len(seg.word) <= 1:
                continue
            elif seg.flag == "eng":
                if len(seg.word) <= 2:
                    continue
                else:
                    filtered_words_list.append(seg.word)
            elif seg.flag.startswith("n"):
                filtered_words_list.append(seg.word)
            elif seg.flag in ["x", "eng"]:  # 是自定一个词语或者是英文单词
                filtered_words_list.append(seg.word)
        return filtered_words_list

    for row in partition:
        news = f"{row.category},{row.title},{row.content},"
        sentence = re.sub("<.*?>", "", news)  # 替换掉标签数据
        words = cut_sentence(sentence)
        yield row.nid, row.category, words


def read_data(spark):

    # 读取表
    data = read_mysql_to_df(spark,table='t_news_analysis')


    # 打印data数据类型
    print(type(data))
    # 展示数据
    data.show()

    words_df = data.rdd.mapPartitions(segmentation).toDF(["article_id", "channel_id", "words"])
    print(words_df.take(5))

    words_df.write.save(WORDS_DF_PATH)

    # calc_word_frequency(words_df)
    #
    # calc_tfidf_model(words_df)
    # 关闭spark会话
    spark.stop()


def calc_word_frequency(words_df):
    # 总词汇的大小，文本中必须出现的次数
    cv = CountVectorizer(inputCol="words", outputCol="countFeatures", vocabSize=200 * 10000, minDF=1.0)
    # 训练词频统计模型
    cv_model = cv.fit(words_df)

    cv_model.write().overwrite().save(COUNT_VECTORIZER_MODEL_PATH)


def calc_tfidf_model(words_df):
    cv_model = CountVectorizerModel.load(COUNT_VECTORIZER_MODEL_PATH)
    # 得出词频向量结果
    cv_result = cv_model.transform(words_df)

    idf = IDF(inputCol="countFeatures", outputCol="idfFeatures")
    idfModel = idf.fit(cv_result)
    idfModel.write().overwrite().save(IDF_MODEL_PATH)

    # cv_model.vocabulary
    print(idfModel.idf.toArray()[:20])


def keyword_func(data):
    for index in range(len(data)):
        data[index] = list(data[index])
        data[index].append(index)
        data[index][1] = float(data[index][1])


def save_keywords_list_with_idf(spark, cv_model, idf_model):
    keywords_list_with_idf = list(zip(cv_model.vocabulary, idf_model.idf.toArray()))

    keyword_func(keywords_list_with_idf)

    sc = spark.sparkContext
    rdd = sc.parallelize(keywords_list_with_idf)

    df = rdd.toDF(["keywords", "idf", "index"])

    df.show()
    save_df_to_mysql(df, table="t_news_idf")


def calc_tfidf(spark):
    words_df = spark.read.parquet(WORDS_DF_PATH)
    cv_model = CountVectorizerModel.load(COUNT_VECTORIZER_MODEL_PATH)
    idf_model = IDFModel.load(IDF_MODEL_PATH)

    cv_result = cv_model.transform(words_df)
    tfidf_result = idf_model.transform(cv_result)

    _keywordsByTFIDF = tfidf_result.rdd.mapPartitions(tfidf_func).toDF(["article_id", "channel_id", "index", "tfidf"])

    _keywordsByTFIDF.show()

    # 利用结果索引与”idf_keywords_values“合并知道词
    keywordsIndex = read_mysql_to_df(spark, table="t_news_idf")
    # keywordsIndex = spark.sql("select keyword, index idx from idf_keywords_values")
    # 利用结果索引与”idf_keywords_values“合并知道词
    keywordsByTFIDF = _keywordsByTFIDF.join(keywordsIndex, keywordsIndex.index == _keywordsByTFIDF.index).select(["article_id", "channel_id", "keywords", "tfidf"])
    # keywordsByTFIDF.write.insertInto("tfidf_keywords_values")

    keywordsByTFIDF.show()
    save_df_to_mysql(keywordsByTFIDF,table=TABLE_TFIDF)



def tfidf_func(partition):
    TOPK = 20
    for row in partition:
        # 找到索引与IDF值并进行排序
        _ = list(zip(row.idfFeatures.indices, row.idfFeatures.values))
        _ = sorted(_, key=lambda x: x[1], reverse=True)
        result = _[:TOPK]
        for word_index, tfidf in result:
            yield row.article_id, row.channel_id, int(word_index), round(float(tfidf), 4)




if __name__ == '__main__':
    spark = get_spark()
    # read_data(spark)
    # df = spark.read.parquet(WORDS_DF_PATH)
    # df.show()

    calc_tfidf(spark)
