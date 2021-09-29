#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : spark_utils.py
# @Author: sl
# @Date  : 2021/9/27 - 下午5:16
from pyspark.sql import SparkSession

from recommend.utils.constants import MYSQL_URL, MYSQL_PROP


def save_df_to_mysql(df, table="t_news_idf"):
    df.write.jdbc(MYSQL_URL, table, mode='overwrite', properties=MYSQL_PROP)


def read_mysql_to_df(spark, table):
    # 读取表
    data = spark.read.jdbc(url=MYSQL_URL, table=table, properties=MYSQL_PROP)
    return data


def read_csv_to_df(spark, path):
    data = spark.read.format('csv').option('header', 'true').load(path)
    return data


def save_df_to_csv(df, path):
    df.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(path)


def show_df(df):
    df.printSchema()
    df.show(10, truncate=False)


def get_spark(appName="sql"):
    spark = SparkSession. \
        Builder(). \
        appName(appName). \
        master('local'). \
        getOrCreate()
    return spark
