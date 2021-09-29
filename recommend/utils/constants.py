#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : constants.py
# @Author: sl
# @Date  : 2021/9/26 - 下午12:15


DATA_THUCNEWS_DIR2 = '/home/sl/workspace/data/nlp/THUCNews'

# NEWS_SUB_DIR = ['时政', '体育', '彩票', '社会', '游戏', '房产', '时尚', '星座', '娱乐', '科技', '家居', '教育']
NEWS_SUB_DIR = ['时政', '体育', '彩票', '社会', '游戏', '房产', '时尚', '星座', '娱乐', '科技', '家居', '教育']

MODEL_BASE_PATH = "file:///home/sl/workspace/data/nlp/recommend_news"

WORDS_DF_PATH = f"{MODEL_BASE_PATH}/models/WORDS_DF.model"
COUNT_VECTORIZER_MODEL_PATH = f"{MODEL_BASE_PATH}/models/CV.model"
IDF_MODEL_PATH = f"{MODEL_BASE_PATH}/models/IDF.model"
WORD2VET_MODEL_PATH = f"{MODEL_BASE_PATH}/models/test11.word2vec"

MYSQL_PROP = {'user': 'root',
              'password': '123456',
              'driver': 'com.mysql.cj.jdbc.Driver'}

MYSQL_URL = 'jdbc:mysql://localhost:3306/re_news'

TABLE_TFIDF = "t_news_keyword_tfidf"
TABLE_TEXT_RANK = "t_news_textrank_keywords_values"
TABLE_ARTICLE_PROFILE = "t_news_article_profile"
TABLE_ARTICLE_VECTOR = "t_news_article_vector"
TABLE_ARTICLE_SIMILAR = "t_news_article_similar"

# 电影推荐 数据集
MOVIE_RESOURCES_PATH = f"{MODEL_BASE_PATH}/sampledata/movies.csv"
RATINGS_RESOURCES_PATH = f"{MODEL_BASE_PATH}/sampledata/ratings.csv"

MOVIE_SAMPLEDATA_PATH = f"{MODEL_BASE_PATH}/sampledata"

MOVIE_RATING_VEC_EMBEDDING_PATH = f"{MOVIE_SAMPLEDATA_PATH}/item2vecEmb.txt"
MOVIE_RATING_GRAPH_EMBEDDING_PATH = f"{MOVIE_SAMPLEDATA_PATH}/itemGraphEmb.txt"
MOVIE_USER_EMBEDDING_PATH = f"{MOVIE_SAMPLEDATA_PATH}/userEmb.txt"
