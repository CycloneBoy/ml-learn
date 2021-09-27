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


def get_spark():
    spark = SparkSession. \
        Builder(). \
        appName('sql'). \
        master('local'). \
        getOrCreate()
    return spark
