#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : FeatureEngForRecModel.py
# @Author: sl
# @Date  : 2021/9/29 - 下午12:49

"""
特征工程

提取电影的特征：播放年 ，平均分，平均方差
提取用户的特征： 观看的前5个用户，用户平均分，用户平方差，用户喜好特征
"""

import pyspark.sql as sql
from pyspark.sql.functions import *
from pyspark.sql.types import *
from collections import defaultdict
from pyspark.sql import functions as F

from recommend.utils.constants import MOVIE_RESOURCES_PATH, RATINGS_RESOURCES_PATH, MODEL_BASE_PATH, \
    MOVIE_SAMPLEDATA_PATH
from recommend.utils.spark_utils import get_spark, read_csv_to_df, save_df_to_csv

NUMBER_PRECISION = 2


def addSampleLabel(ratingSamples):
    ratingSamples.show(5, truncate=False)
    ratingSamples.printSchema()
    sampleCount = ratingSamples.count()
    ratingSamples.groupBy('rating').count().orderBy('rating').withColumn('percentage',
                                                                         F.col('count') / sampleCount).show()
    ratingSamples = ratingSamples.withColumn('label', when(F.col('rating') >= 3.5, 1).otherwise(0))
    return ratingSamples


def extractReleaseYearUdf(title):
    if not title or len(title.strip()) < 6:
        return 1998
    else:
        yearStr = title.strip()[-5:-1]
    return int(yearStr)


def addMovieFeatures(movieSamples, ratingSamplesWithLabel):
    # add movie basic features
    samplesWithMovies1 = ratingSamplesWithLabel.join(movieSamples, on=['movieId'], how='left')
    # add releaseYear,title
    samplesWithMovies2 = samplesWithMovies1.withColumn('releaseYear',
                                                       udf(extractReleaseYearUdf, IntegerType())('title')) \
        .withColumn('title', udf(lambda x: x.strip()[:-6].strip(), StringType())('title')) \
        .drop('title')

    samplesWithMovies2.show(truncate=False)

    # split genres
    samplesWithMovies3 = samplesWithMovies2.withColumn('movieGenre1', split(F.col('genres'), "\\|")[0]) \
        .withColumn('movieGenre2', split(F.col('genres'), "\\|")[1]) \
        .withColumn('movieGenre3', split(F.col('genres'), "\\|")[2])

    samplesWithMovies3.show(truncate=False)
    # add rating features
    movieRatingFeatures = samplesWithMovies3.groupBy('movieId') \
        .agg(F.count(F.lit(1)).alias('movieRatingCount'),
             format_number(F.avg(F.col('rating')), NUMBER_PRECISION).alias('movieAvgRating'),
             F.stddev(F.col('rating')).alias('movieRatingStddev')).fillna(0) \
        .withColumn('movieRatingStddev', format_number(F.col('movieRatingStddev'), NUMBER_PRECISION))

    movieRatingFeatures.show(truncate=False)

    # join movie rating features
    samplesWithMovies4 = samplesWithMovies3.join(movieRatingFeatures, on=['movieId'], how='left')
    samplesWithMovies4.printSchema()
    samplesWithMovies4.show(5, truncate=False)
    return samplesWithMovies4


def extractGenres(genres_list):
    '''
    pass in a list which format like ["Action|Adventure|Sci-Fi|Thriller", "Crime|Horror|Thriller"]
    count by each genre，return genre_list in reverse order
    eg:
    (('Thriller',2),('Action',1),('Sci-Fi',1),('Horror', 1), ('Adventure',1),('Crime',1))
    return:['Thriller','Action','Sci-Fi','Horror','Adventure','Crime']
    '''
    genres_dict = defaultdict(int)
    for genres in genres_list:
        for genre in genres.split('|'):
            genres_dict[genre] += 1
    sortedGenres = sorted(genres_dict.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sortedGenres]


def addUserFeatures(samplesWithMovieFeatures):
    extractGenresUdf = udf(extractGenres, ArrayType(StringType()))
    samplesWithUserFeatures = samplesWithMovieFeatures \
        .withColumn('userPositiveHistory',
                    F.collect_list(when(F.col('label') == 1, F.col('movieId')).otherwise(F.lit(None))).over(
                        sql.Window.partitionBy("userId").orderBy(F.col("timestamp")).rowsBetween(-100, -1))) \
        .withColumn("userPositiveHistory", reverse(F.col("userPositiveHistory"))) \
        .withColumn('userRatedMovie1', F.col('userPositiveHistory')[0]) \
        .withColumn('userRatedMovie2', F.col('userPositiveHistory')[1]) \
        .withColumn('userRatedMovie3', F.col('userPositiveHistory')[2]) \
        .withColumn('userRatedMovie4', F.col('userPositiveHistory')[3]) \
        .withColumn('userRatedMovie5', F.col('userPositiveHistory')[4]) \
        .withColumn('userRatingCount',
                    F.count(F.lit(1)).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn('userAvgReleaseYear', F.avg(F.col('releaseYear')).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)).cast(IntegerType())) \
        .withColumn('userReleaseYearStddev', F.stddev(F.col("releaseYear")).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn("userAvgRating", format_number(
        F.avg(F.col("rating")).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)),
        NUMBER_PRECISION)) \
        .withColumn("userRatingStddev", F.stddev(F.col("rating")).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn("userGenres", extractGenresUdf(
        F.collect_list(when(F.col('label') == 1, F.col('genres')).otherwise(F.lit(None))).over(
            sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)))) \
        .withColumn("userRatingStddev", format_number(F.col("userRatingStddev"), NUMBER_PRECISION)) \
        .withColumn("userReleaseYearStddev", format_number(F.col("userReleaseYearStddev"), NUMBER_PRECISION)) \
        .withColumn("userGenre1", F.col("userGenres")[0]) \
        .withColumn("userGenre2", F.col("userGenres")[1]) \
        .withColumn("userGenre3", F.col("userGenres")[2]) \
        .withColumn("userGenre4", F.col("userGenres")[3]) \
        .withColumn("userGenre5", F.col("userGenres")[4]) \
        .drop("genres", "userGenres", "userPositiveHistory") \
        .filter(F.col("userRatingCount") > 1)
    samplesWithUserFeatures.printSchema()
    samplesWithUserFeatures.show(10, truncate=False)
    samplesWithUserFeatures.filter(samplesWithMovieFeatures['userId'] == 1).orderBy(F.col('timestamp').asc()).show(
        truncate=False)
    return samplesWithUserFeatures


def splitAndSaveTrainingTestSamples(samplesWithUserFeatures, file_path):
    smallSamples = samplesWithUserFeatures.sample(0.1)
    training, test = smallSamples.randomSplit((0.8, 0.2))
    trainingSavePath = file_path + '/trainingSamples'
    testSavePath = file_path + '/testSamples'

    save_df_to_csv(training, trainingSavePath)
    save_df_to_csv(test, testSavePath)


def splitAndSaveTrainingTestSamplesByTimeStamp(samplesWithUserFeatures, file_path):
    smallSamples = samplesWithUserFeatures.sample(0.1).withColumn("timestampLong", F.col("timestamp").cast(LongType()))
    quantile = smallSamples.stat.approxQuantile("timestampLong", [0.8], 0.05)
    splitTimestamp = quantile[0]
    training = smallSamples.where(F.col("timestampLong") <= splitTimestamp).drop("timestampLong")
    test = smallSamples.where(F.col("timestampLong") > splitTimestamp).drop("timestampLong")
    trainingSavePath = file_path + '/trainingSamples'
    testSavePath = file_path + '/testSamples'

    save_df_to_csv(training, trainingSavePath)
    save_df_to_csv(test, testSavePath)


if __name__ == '__main__':
    spark = get_spark()
    movieSamples = read_csv_to_df(spark, path=MOVIE_RESOURCES_PATH)
    movieSamples.show()
    ratingSamples = read_csv_to_df(spark, path=RATINGS_RESOURCES_PATH)
    ratingSamplesWithLabel = addSampleLabel(ratingSamples)
    ratingSamplesWithLabel.show(10, truncate=False)

    samplesWithMovieFeatures = addMovieFeatures(movieSamples, ratingSamplesWithLabel)
    samplesWithUserFeatures = addUserFeatures(samplesWithMovieFeatures)
    # # save samples as csv format
    splitAndSaveTrainingTestSamples(samplesWithUserFeatures, MOVIE_SAMPLEDATA_PATH)
    # splitAndSaveTrainingTestSamplesByTimeStamp(samplesWithUserFeatures, MOVIE_SAMPLEDATA_PATH)
