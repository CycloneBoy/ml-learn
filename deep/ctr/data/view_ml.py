#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : view_ml.py
# @Author: sl
# @Date  : 2021/9/18 - 下午6:12

"""
探索数据
"""

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    conf = SparkConf().setAppName("miniProject").setMaster("local[*]")
    sc = SparkContext.getOrCreate(conf)
    sentencesRDD = sc.parallelize(['Hello world', 'My name is Patrick'])
    wordsRDD = sentencesRDD.flatMap(lambda sentence: sentence.split(" "))
    print(wordsRDD.collect())
    print(wordsRDD.count())

    lines = sc.textFile("file:///home/sl/workspace/python/a2020/ml-learn/deep/ctr/data/view_ml.py")
    print(lines.first())

    # # # spark is an existing SparkSession
    # df = sc. ("/home/sl/workspace/data/nlp/voidful/albert_chinese_tiny/config.json")
    # # Displays the content of the DataFrame to stdout
    # df.show()
