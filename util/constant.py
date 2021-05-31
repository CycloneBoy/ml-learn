#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : constant.py
# @Author: sl
# @Date  : 2020/9/17 - 下午10:17
import os

WORK_DIR = "/home/sl/workspace/python/a2020/ml-learn"

LOG_DIR = os.path.join(WORK_DIR, "data/log")

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

DATA_MNIST_DIR = '~/workspace/data/mnist'
MODEL_MNIST_DIR = '{}/models'.format(DATA_MNIST_DIR)

DATA_FASHION_MNIST_DIR = '~/workspace/data/fashionmnist'

DATA_CACHE_DIR = '/home/sl/workspace/data/nlp'

GLOVE_DATA_DIR = '/home/sl/workspace/data/nlp/glove/glove.6B'

# aclImdb_v1.tar.gz
IMDB_DATA_DIR = '/home/sl/workspace/data/nlp/aclImdb'

# NLP模型保存
MODEL_NLP_DIR = '/home/sl/workspace/data/nlp/model'

# opencv 图片地址
OPENCV_IMAGE_DIR = '/home/sl/data'
BILIBILI_VIDEO_IMAGE_DIR = '/home/sl/data/bilibili'

############################################################################
# 文本相关的数据路径

# 新闻语料库
DATA_TXT_NEWS_DIR = '/home/sl/workspace/python/github/learning-nlp-master/chapter-3/data/news'
# 停用词路径 合并github上的5个文件后的,停用词大小: 2524
DATA_TXT_STOP_WORDS_DIR = '/home/sl/workspace/data/nlp/stopwords/stop_words.utf8'
# github 上的停用词
DATA_TXT_STOP_WORDS_GITHUB_DIR = '/home/sl/workspace/data/nlp/stopwords'

# THUCNews 路径
DATA_THUCNEWS_DIR = '/home/sl/workspace/python/a2020/ml-learn/data/nlp/THUCNews'

# sgns.sogou.char  词嵌入
DATA_EMBEDDING_SOGOU_CHAR = "/home/sl/workspace/data/nlp/sgns.sogou.char"

NLP_PRETRAIN_DIR = DATA_CACHE_DIR
# BERT_BASE_CHINESE = '/home/sl/workspace/data/nlp/bert-base-chinese'
############################################################################

# 爬虫html页面
DATA_HTML_DIR = os.path.join(WORK_DIR, "data/txt/html")

# 爬虫问题保存地址
DATA_QUESTION_DIR = os.path.join(WORK_DIR, "data/txt/result")

# 爬虫问题保存地址 JSON
DATA_JSON_DIR = os.path.join(WORK_DIR, "data/txt/json")

# 问答
DATA_QUESTION_ANSWER_DIR = os.path.join(DATA_CACHE_DIR, "question/result")

QA_DELIMITER = "|"