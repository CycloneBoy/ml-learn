#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : textrank_sklearn.py
# @Author: sl
# @Date  : 2021/7/3 -  下午12:52

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import jieba
import re

from util.constant import TEST_SUMMARY_1, TEST_SUMMARY_DOC
from util.nlp_utils import cut_sentence


def tfidf_sim(sentences):
    """
     tfidf相似度
    :param sentences:
    :return:
    """
    # tfidf计算
    model = TfidfVectorizer(tokenizer=jieba.cut,
                            ngram_range=(1, 2), # 3,5
                            stop_words=[' ', '\t', '\n'],  # 停用词
                            max_features=10000,
                            token_pattern=r"(?u)\b\w+\b",  # 过滤停用词
                            min_df=1,
                            max_df=0.9,
                            use_idf=1,  # 光滑
                            smooth_idf=1,  # 光滑
                            sublinear_tf=1, )  # 光滑
    matrix = model.fit_transform(sentences)
    matrix_norm = TfidfTransformer().fit_transform(matrix)
    return matrix_norm


def textrank_tfidf1(sentences,topK=6):
    """
        使用tf-idf作为相似度, networkx.pagerank获取中心句子作为摘要
    :param sentences: str, docs of text
    :param topk:int
    :return:list
    """
    # 切句子
    sentences = list(cut_sentence(sentences))
    print("分句子：{}".format(len(sentences)))
    # tf-idf相似度
    matrix_norm = tfidf_sim(sentences)
    # 构建相似度矩阵
    tfidf_sim_matrix = nx.from_scipy_sparse_matrix(matrix_norm * matrix_norm.T)
    # nx.pagerank
    sens_scores = nx.pagerank(tfidf_sim_matrix)
    # 得分排序
    sen_rank = sorted(sens_scores.items(),key=lambda x: x[1],reverse=True)
    # 保留topk个, 防止越界
    topk = min(len(sentences),topK)
    # 返回原句子和得分
    return [(sr[1],sentences[sr[0]]) for sr in sen_rank][0:topk]


if __name__ == '__main__':

    sen = TEST_SUMMARY_1[1]
    # sen = TEST_SUMMARY_DOC

    res1 = cut_sentence(sen)

    print(res1)


    for score_sen in textrank_tfidf1(sen,32):
        print(score_sen)