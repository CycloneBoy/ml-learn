#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : Embedding.py
# @Author: sl
# @Date  : 2021/9/29 - 下午3:40

"""
Embedding 嵌入

1. item2Vec
根据用户看过的电影列表 构建 电影的嵌入向量
（ 用户看过的电影按照时间排序，代表一个句子，多个用户代表多个句子，然后训练 词向量(一部电影代表一个词)）


2. graph2Vec
    DeepWalk 算法
         用户看过的电影按照时间排序，代表一个句子，然后采用2-gram 采样 为多条 边 ，
         根据边 来构造网络 （权重节点 和边的权重）
         随机游走模型进行 采样 序列 ，代表一个句子
         最后进行训练词向量

3. user2Vec
    根据得到的 item2Vec ,编码 一个用户看过的所有电影的 平均值 即为当前用户的向量 （有点像 doc2vec）


"""

import os

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.mllib.feature import Word2Vec
from pyspark.ml.linalg import Vectors
import random
from collections import defaultdict
import numpy as np
from pyspark.sql import functions as F

from recommend.utils.constants import RATINGS_RESOURCES_PATH, MOVIE_RATING_VEC_EMBEDDING_PATH, \
    MOVIE_RATING_VEC_EMBEDDING_PATH, MOVIE_USER_EMBEDDING_PATH, MOVIE_RATING_GRAPH_EMBEDDING_PATH
from recommend.utils.spark_utils import get_spark, read_csv_to_df, show_df


class UdfFunction:
    @staticmethod
    def sortF(movie_list, timestamp_list):
        """
        sort by time and return the corresponding movie sequence
        eg:
            input: movie_list:[1,2,3]
                   timestamp_list:[1112486027,1212546032,1012486033]
            return [3,1,2]
        """
        pairs = []
        for m, t in zip(movie_list, timestamp_list):
            pairs.append((m, t))
        # sort by time
        pairs = sorted(pairs, key=lambda x: x[1])
        return [x[0] for x in pairs]


def processItemSequence(ratingSamples):
    sortUdf = udf(UdfFunction.sortF, ArrayType(StringType()))
    userSeq = ratingSamples.where(F.col("rating") >= 3.5) \
        .groupBy("userId") \
        .agg(sortUdf(F.collect_list("movieId"), F.collect_list("timestamp")).alias('movieIds')) \
        .withColumn("movieIdStr", array_join(F.col("movieIds"), " "))
    # userSeq.select("userId", "movieIdStr").show(10, truncate = False)

    show_df(userSeq)
    return userSeq.select('movieIdStr').rdd.map(lambda x: x[0].split(' '))


def embeddingLSH(spark, movieEmbMap):
    # read word2Vec
    movieEmbSeq = []
    for key, embedding_list in movieEmbMap.items():
        embedding_list = [np.float64(embedding) for embedding in embedding_list]
        movieEmbSeq.append((key, Vectors.dense(embedding_list)))
    movieEmbDF = spark.createDataFrame(movieEmbSeq).toDF("movieId", "emb")
    show_df(movieEmbDF)

    # LSH hash find  5 nearest neighbors
    bucketProjectionLSH = BucketedRandomProjectionLSH(inputCol="emb", outputCol="bucketId", bucketLength=0.1,
                                                      numHashTables=3)
    bucketModel = bucketProjectionLSH.fit(movieEmbDF)
    embBucketResult = bucketModel.transform(movieEmbDF)

    show_df(embBucketResult)

    print("Approximately searching for 5 nearest neighbors of the sample embedding:")
    sampleEmb = Vectors.dense(0.795, 0.583, 1.120, 0.850, 0.174, -0.839, -0.0633, 0.249, 0.673, -0.237)
    bucketModel.approxNearestNeighbors(movieEmbDF, sampleEmb, 5).show(truncate=False)


def trainItem2vec(spark, samples, embLength, embOutputPath, saveToRedis, redisKeyPrefix):
    # train word2Vec
    word2Vec = Word2Vec().setVectorSize(embLength).setWindowSize(5).setNumIterations(10)
    model = word2Vec.fit(samples)

    # find same word like "158"
    synonyms = model.findSynonyms("158", 20)
    for synonym, cosineSimilarity in synonyms:
        print(synonym, cosineSimilarity)

    embOutputDir = '/'.join(embOutputPath.split('/')[:-1])
    if not os.path.exists(embOutputDir):
        os.makedirs(embOutputDir)

    print(f"save embedding file :{embOutputPath}")
    # save word2Vec to file
    with open(embOutputPath, 'w', encoding='utf-8') as f:
        for movie_id in model.getVectors():
            vectors = " ".join([str(emb) for emb in model.getVectors()[movie_id]])
            f.write(movie_id + ":" + vectors + "\n")

    embeddingLSH(spark, model.getVectors())
    return model


def generate_pair(x):
    # eg:
    # watch sequence:['858', '50', '593', '457']
    # return:[['858', '50'],['50', '593'],['593', '457']]
    pairSeq = []
    previousItem = ''
    for item in x:
        if not previousItem:
            previousItem = item
        else:
            pairSeq.append((previousItem, item))
            previousItem = item
    return pairSeq


def generate_pair2(x):
    return [[x[i], x[i + 1]] for i in range(len(x) - 1)]


def generateTransitionMatrix(samples):
    # user see the movie list order by time pair
    pairSamples = samples.flatMap(lambda x: generate_pair(x))
    pairCountMap = pairSamples.countByValue()
    pairTotalCount = 0
    transitionCountMatrix = defaultdict(dict)
    itemCountMap = defaultdict(int)
    for key, cnt in pairCountMap.items():
        key1, key2 = key
        # 一条边的权重
        transitionCountMatrix[key1][key2] = cnt
        itemCountMap[key1] += cnt
        pairTotalCount += cnt
    transitionMatrix = defaultdict(dict)
    itemDistribution = defaultdict(dict)

    # 归一化 边的权值
    for key1, transitionMap in transitionCountMatrix.items():
        for key2, cnt in transitionMap.items():
            transitionMatrix[key1][key2] = transitionCountMatrix[key1][key2] / itemCountMap[key1]

    # 归一化 节点的的权值
    for itemid, cnt in itemCountMap.items():
        itemDistribution[itemid] = cnt / pairTotalCount

    print("transitionMatrix")
    # print(transitionMatrix)
    print("itemDistribution")
    # print(itemDistribution)

    return transitionMatrix, itemDistribution


def oneRandomWalk(transitionMatrix, itemDistribution, sampleLength):
    """
    随机游走模型：
    一种可重复访问已经访问节点的深度优先遍历算法

    :param transitionMatrix:
    :param itemDistribution:
    :param sampleLength:
    :return:
    """
    sample = []
    randomDouble = random.random()

    # pick the first element
    firstItem = ""
    accumulateProb = 0.0
    for item, prob in itemDistribution.items():
        accumulateProb += prob
        if accumulateProb >= randomDouble:
            firstItem = item
            break
    sample.append(firstItem)
    curElement = firstItem
    i = 1

    # 从当前节点 随机按照边的权重采样 sampleLength 个节点
    while i < sampleLength:
        if (curElement not in itemDistribution) or (curElement not in transitionMatrix):
            break
        probDistribution = transitionMatrix[curElement]
        randomDouble = random.random()
        accumulateProb = 0.0
        for item, prob in probDistribution.items():
            accumulateProb += prob
            if accumulateProb >= randomDouble:
                curElement = item
                break
        sample.append(curElement)
        i += 1
    return sample


def randomWalk(transitionMatrix, itemDistribution, sampleCount, sampleLength):
    samples = []
    for i in range(sampleCount):
        samples.append(oneRandomWalk(transitionMatrix, itemDistribution, sampleLength))
    return samples


def graphEmb(samples, spark, embLength, embOutputFilename, saveToRedis, redisKeyPrefix):
    transitionMatrix, itemDistribution = generateTransitionMatrix(samples)
    sampleCount = 20000
    sampleLength = 10
    newSamples = randomWalk(transitionMatrix, itemDistribution, sampleCount, sampleLength)

    # 构造rdd
    rddSamples = spark.sparkContext.parallelize(newSamples)
    trainItem2vec(spark, rddSamples, embLength, embOutputFilename, saveToRedis, redisKeyPrefix)


def generateUserEmb(spark, ratingSamples, model, embLength, embOutputPath, saveToRedis, redisKeyPrefix):
    Vectors_list = []
    for key, value in model.getVectors().items():
        Vectors_list.append((key, list(value)))

    fields = [
        StructField('movieId', StringType(), False),
        StructField('emb', ArrayType(FloatType()), False)
    ]

    schema = StructType(fields)
    Vectors_df = spark.createDataFrame(Vectors_list, schema=schema)
    ratingSamples = ratingSamples.join(Vectors_df, on='movieId', how='inner')

    # 对每个用户看过的电影的 词向量 平均 就是当前用户的 嵌入向量
    result = ratingSamples.select('userId', 'emb').rdd.map(lambda x: (x[0], x[1])) \
        .reduceByKey(lambda a, b: [a[i] + b[i] for i in range(len(a))]).collect()

    with open(embOutputPath, 'w', encoding='utf-8') as f:
        for row in result:
            vectors = " ".join([str(emb) for emb in row[1]])
            f.write(row[0] + ":" + vectors + "\n")


if __name__ == '__main__':
    spark = get_spark()

    # rating data
    ratingSamples = read_csv_to_df(spark, path=RATINGS_RESOURCES_PATH)
    show_df(ratingSamples)

    # same userId see the movie list
    samples = processItemSequence(ratingSamples)

    samples.show()

    embLength = 10
    model = trainItem2vec(spark, samples, embLength,
                          embOutputPath=MOVIE_RATING_VEC_EMBEDDING_PATH[7:], saveToRedis=False,
                          redisKeyPrefix="i2vEmb")

    graphEmb(samples, spark, embLength, embOutputFilename=MOVIE_RATING_GRAPH_EMBEDDING_PATH[7:],
             saveToRedis=True, redisKeyPrefix="graphEmb")

    generateUserEmb(spark, ratingSamples, model, embLength,
                    embOutputPath=MOVIE_USER_EMBEDDING_PATH[7:], saveToRedis=False,
                    redisKeyPrefix="uEmb")
