#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : build_news_word2vet.py
# @Author: sl
# @Date  : 2021/9/27 - 下午10:04


"""
在新闻推荐中有很多地方需要推荐相似文章，包括首页频道可以推荐相似的文章，详情页猜你喜欢

需求

首页频道推荐：每个频道推荐的时候，会通过计算两两文章相似度，
        快速达到在线推荐的效果，比如用户点击文章，我们可以将离线计算好相似度的文章排序快速推荐给该用户。此方式也就可以解决冷启动的问题
方式：
1、计算两两文章TFIDF之间的相似度
2、计算两两文章的word2vec或者doc2vec向量相似度
我们采用第二种方式，实践中word2vec在大量数据下达到的效果更好
"""
import json

from pyspark.ml.feature import Word2Vec, Word2VecModel, BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors

from recommend.utils.constants import WORDS_DF_PATH, WORD2VET_MODEL_PATH, TABLE_ARTICLE_PROFILE, TABLE_ARTICLE_VECTOR, \
    TABLE_ARTICLE_SIMILAR
from recommend.utils.spark_utils import get_spark, read_mysql_to_df, save_df_to_mysql


def build_word2vet(spark):
    words_df = spark.read.parquet(WORDS_DF_PATH)
    words_df.show()

    new_word2Vec = Word2Vec(vectorSize=100, inputCol="words", outputCol="model", minCount=3)
    new_model = new_word2Vec.fit(words_df)
    new_model.save(WORD2VET_MODEL_PATH)


def load_word2vet(spark):
    wv_model = Word2VecModel.load(WORD2VET_MODEL_PATH)
    vectors = wv_model.getVectors()
    vectors.show()

    profile = read_mysql_to_df(spark, table=TABLE_ARTICLE_PROFILE)
    profile = profile.rdd.map(change_to_map).toDF(["article_id", "channel_id", "keywords", "topics"])
    profile.registerTempTable("profile")

    incremental = spark.sql("select * from profile")
    incremental.registerTempTable("incremental")

    articleKeywordsWeights = spark.sql(
        "select article_id, channel_id, keyword, weight from incremental LATERAL VIEW explode(keywords) AS keyword, weight")
    _article_profile = articleKeywordsWeights.join(vectors, vectors.word == articleKeywordsWeights.keyword, "inner")

    articleKeywordVectors = _article_profile.rdd.map(
        lambda row: (row.article_id, row.channel_id, row.keyword, row.weight * row.vector)) \
        .toDF(["article_id", "channel_id", "keyword", "weightingVector"])

    articleKeywordVectors.registerTempTable("tempTable")
    articleVector = spark.sql(
        "select article_id, min(channel_id) channel_id, collect_set(weightingVector) vectors from tempTable group by article_id") \
        .rdd.map(avg_func).toDF(["article_id", "channel_id", "articleVector"])

    articleVector = articleVector.rdd.map(toArray).toDF(['article_id', 'channel_id', 'articleVector'])

    articleVector.show()
    save_df_to_mysql(articleVector, table=TABLE_ARTICLE_VECTOR)


def avg_func(row):
    x = 0
    for v in row.vectors:
        x += v
    #  将平均向量作为article的向量
    return row.article_id, row.channel_id, x / len(row.vectors)


def toArray(row):
    articleVector = [str(float(i)) for i in row.articleVector.toArray()]
    articleVector_str = "|".join(articleVector)
    return row.article_id, row.channel_id, articleVector_str


def change_to_map(row):
    keyword = json.loads(row.keywords)
    topic = str(row.topics).split("|")
    return row.article_id, row.channel_id, keyword, topic


def load_news_vector(spark):
    """读取 word2vec"""
    article_vector_df = read_mysql_to_df(spark, table=TABLE_ARTICLE_VECTOR)
    train = article_vector_df.rdd.map(_array_to_vector).toDF(['article_id', 'channel_id', 'articleVector'])

    brp = BucketedRandomProjectionLSH(inputCol='articleVector',
                                      outputCol='hashes', numHashTables=4.0, bucketLength=10.0)
    model = brp.fit(train)

    similar = model.approxSimilarityJoin(train, train, 2.0, distCol='EuclideanDistance')

    similar_res = similar.sort(['EuclideanDistance'], ascending=False)
    similar_res.show()

    sim_data_list = similar_res.rdd.map(build_sim_result_func).toDF(
        ['article_id', 'channel_id', 'sim_article_id', 'distance'])

    sim_data_list = sim_data_list.rdd.filter(filter_same_id).toDF(
        ['article_id', 'channel_id', 'sim_article_id', 'distance'])
    sim_data_list.show()

    save_df_to_mysql(sim_data_list,table=TABLE_ARTICLE_SIMILAR)


def _array_to_vector(row):
    articleVector_list = str(row.articleVector).split("|")
    articleVector = [float(vec) for vec in articleVector_list]
    return row.article_id, row.channel_id, Vectors.dense(articleVector)


def save_to_mysql(partition):
    data_list = []
    for row in partition:
        if row.datasetA.article_id == row.datasetB.article_id:
            pass
        else:
            article_id = row.datasetA.article_id
            cid = row.datasetA.channel_id
            sim_id = row.datasetB.article_id
            sim_score = row.EuclideanDistance
            data_list.append((article_id, cid, sim_id, sim_score))

    return data_list


def build_sim_result_func(row):
    article_id = row.datasetA.article_id
    cid = row.datasetA.channel_id
    sim_id = row.datasetB.article_id
    sim_score = row.EuclideanDistance
    return article_id, cid, sim_id, sim_score


def filter_same_id(row):
    article_id = row.article_id
    sim_article_id = row.sim_article_id
    return article_id != sim_article_id


if __name__ == '__main__':
    spark = get_spark()
    # build_word2vet(spark)
    # load_word2vet(spark)
    load_news_vector(spark)
