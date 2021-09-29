#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : FeatureEngineering.py
# @Author: sl
# @Date  : 2021/9/29 - 下午2:53

"""
特征工程：
one-hot , multi-hot , numerical-features (连续特征 分桶（100）,最大最小归一化)
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, QuantileDiscretizer, MinMaxScaler
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F

from recommend.utils.constants import MOVIE_RESOURCES_PATH, RATINGS_RESOURCES_PATH
from recommend.utils.spark_utils import get_spark, read_csv_to_df, show_df


def oneHotEncoderExample(movieSamples):
    samplesWithIdNumber = movieSamples.withColumn("movieIdNumber", F.col('movieId').cast(IntegerType()))
    encoder = OneHotEncoderEstimator(inputCols=["movieIdNumber"], outputCols=['movieIdVector'], dropLast=False)
    oneHotEncoderSamples = encoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    show_df(oneHotEncoderSamples)


def array2vec(genreIndexes, indexSize):
    genreIndexes.sort()
    fill_list = [1.0 for _ in range(len(genreIndexes))]
    return Vectors.sparse(indexSize, genreIndexes, fill_list)


def multiHotEncoderExample(movieSamples):
    show_df(movieSamples)
    samplesWithGenre = movieSamples.select("movieId", "title", explode(
        split(F.col("genres"), "\\|").cast(ArrayType(StringType()))).alias('genre'))
    show_df(samplesWithGenre)

    genreIndexer = StringIndexer(inputCol='genre', outputCol="genreIndex")
    StringIndexerModel = genreIndexer.fit(samplesWithGenre)
    genreIndexSamples = StringIndexerModel.transform(samplesWithGenre) \
        .withColumn("genreIndexInt", F.col("genreIndex").cast(IntegerType()))
    show_df(genreIndexSamples)

    indexSize = genreIndexSamples.agg(max(F.col("genreIndexInt"))).head()[0] + 1
    processedSamples = genreIndexSamples.groupBy('movieId').agg(
        F.collect_list('genreIndexInt').alias('genreIndexes')).withColumn("indexSize", F.lit(indexSize))
    show_df(processedSamples)

    finalSample = processedSamples.withColumn("vector",
                                              udf(array2vec, VectorUDT())(F.col("genreIndexes"), F.col("indexSize")))
    show_df(finalSample)


def ratingFeatures(ratingSamples):
    show_df(ratingSamples)
    # calculate average movie rating score and rating count
    movieFeatures = ratingSamples.groupBy('movieId').agg(F.count(F.lit(1)).alias('ratingCount'),
                                                         F.avg("rating").alias("avgRating"),
                                                         F.variance('rating').alias('ratingVar')) \
        .withColumn('avgRatingVec', udf(lambda x: Vectors.dense(x), VectorUDT())('avgRating'))

    show_df(movieFeatures)

    # bucketing
    ratingCountDiscretizer = QuantileDiscretizer(numBuckets=100, inputCol="ratingCount", outputCol="ratingCountBucket")
    # Normalization
    ratingScaler = MinMaxScaler(inputCol="avgRatingVec", outputCol="scaleAvgRating")
    pipelineStage = [ratingCountDiscretizer, ratingScaler]
    featurePipeline = Pipeline(stages=pipelineStage)
    movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)
    show_df(movieProcessedFeatures)


if __name__ == '__main__':
    spark = get_spark()
    movieSamples = read_csv_to_df(spark, path=MOVIE_RESOURCES_PATH)
    print("Raw Movie Samples:")
    show_df(movieSamples)

    print("OneHotEncoder Example:")
    oneHotEncoderExample(movieSamples)
    print("MultiHotEncoder Example:")
    multiHotEncoderExample(movieSamples)
    print("Numerical features Example:")

    ratingSamples = read_csv_to_df(spark, path=RATINGS_RESOURCES_PATH)
    ratingFeatures(ratingSamples)
