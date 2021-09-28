#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : build_news_keyword.py
# @Author: sl
# @Date  : 2021/9/27 - 下午8:56


"""
新闻关键词提取
1、加载IDF，保留关键词以及权重计算(TextRank * IDF)
2、合并关键词权重到字典结果
3、将tfidf和textrank共现的词作为主题词
4、将主题词表和关键词表进行合并，插入表

增量更新
问题：计算出TFIDF，TF文档词频，IDF 逆文档频率（文档数量、某词出现的文档数量）
    已有N个文章中词的IDF会随着新增文章而动态变化，就会涉及TFIDF的增量计算。
解决办法：可以在固定时间定时对所有文章数据进行全部计算CV和IDF的模型结果，替换模型即可

"""
import json

from recommend.utils.constants import TABLE_TFIDF, TABLE_TEXT_RANK, TABLE_ARTICLE_PROFILE
from recommend.utils.spark_utils import get_spark, read_mysql_to_df, save_df_to_mysql


def build_keyword(spark):
    idf = read_mysql_to_df(spark, table=TABLE_TFIDF)
    idf.registerTempTable("tfidf_keywords_values")

    idf = idf.withColumnRenamed("keywords", "keyword1")
    idf = idf.withColumnRenamed("article_id", "aid")
    idf = idf.withColumnRenamed("channel_id", "cid")

    textrank_keywords_df = read_mysql_to_df(spark, table=TABLE_TEXT_RANK)
    textrank_keywords_df.registerTempTable("textrank_keywords_values")

    result = textrank_keywords_df.join(idf, textrank_keywords_df.keyword == idf.keyword1)
    keywords_res = result.withColumn("weights", result.textrank * result.tfidf).select(
        ["article_id", "channel_id", "keyword", "weights"])

    keywords_res.registerTempTable("temptable")
    merge_keywords = spark.sql(
        "select article_id, min(channel_id) channel_id, collect_list(keyword) keywords, collect_list(weights) weights from temptable group by article_id")

    keywords_info = merge_keywords.rdd.map(build_row_func).toDF(["article_id", "channel_id", "keywords"])
    keywords_info.show()

    topic_sql = """
                    select t.article_id article_id2, collect_set(t.keywords) topics from tfidf_keywords_values t
                    inner join 
                    textrank_keywords_values r
                    where t.keywords=r.keyword
                    group by article_id2
                    """
    article_topics = spark.sql(topic_sql)
    article_topics.show()

    article_profile = keywords_info.join(article_topics, keywords_info.article_id == article_topics.article_id2) \
        .select(["article_id", "channel_id", "keywords", "topics"])
    article_profile.show()

    article_profile = article_profile.rdd.map(build_row_to_json_func).toDF(
        ["article_id", "channel_id", "keywords", "topics"])
    save_df_to_mysql(article_profile, TABLE_ARTICLE_PROFILE)


# 合并关键词权重合并成字典
def build_row_func(row):
    return row.article_id, row.channel_id, dict(zip(row.keywords, row.weights))


# 转义对应的字符
def build_row_to_json_func(row):
    keyword = json.dumps(row.keywords, ensure_ascii=False)
    topic = "|".join(row.topics)
    return row.article_id, row.channel_id, keyword, topic


if __name__ == '__main__':
    spark = get_spark()
    build_keyword(spark)
