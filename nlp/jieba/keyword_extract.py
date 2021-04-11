#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : keyword_extract.py
# @Author: sl
# @Date  : 2021/4/11 -  下午10:23


"""
主题关键词
- TF-IDF
- TextRank
- LSA
- LDA

"""
import math

from gensim import corpora, models
import jieba
import jieba.posseg as psg
from jieba import analyse
import functools
import numpy as np

from util.common_utils import get_TF
from util.file_utils import get_news_path, get_content, save_to_text
from util.logger_utils import get_log
import os

from util.nlp_utils import stop_words, seg_to_list

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))


def word_filter(seg_list, pos=False):
    """
    去除干扰词
    :param seg_list:
    :param pos:
    :return:
    """
    stop_words_list = stop_words()
    filter_list = []
    # 根据POS参数选择是否词性过滤
    ## 不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        # 过滤停用词表中的词，以及长度为<2的词
        if not word in stop_words_list and len(word) > 1:
            filter_list.append(word)

    return filter_list


def load_data(pos=False, corpus_path='../../data/nlp/news_corpus.txt'):
    """
    数据加载，pos为是否词性标注的参数，corpus_path为数据集路径
    :param pos:
    :param corpus_path:
    :return:
    """
    doc_list = []
    for line in open(corpus_path, 'r', encoding='utf-8'):
        content = line.strip()
        seg_list = seg_to_list(content, pos)
        filter_list = word_filter(seg_list, pos)
        doc_list.append(filter_list)

    return doc_list


def train_idf(doc_list):
    """
    idf值统计方法
    :param doc_list:
    :return:
    """
    idf_dic = {}
    # 总文档数
    doc_count = len(doc_list)

    # 每个词出现的文档数
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0

    # 按公式转换为idf值，分母加1进行平滑处理
    for k, v in idf_dic.items():
        idf_dic[k] = math.log(doc_count / (1.0 + v))

    # 对于没有在字典中的词，默认其仅在一个文档出现，得到默认idf值
    default_idf = math.log(doc_count / (1.0))
    return idf_dic, default_idf


def cmp(e1, e2):
    """
    排序函数，用于topK关键词的按值排序
    :param e1:
    :param e2:
    :return:
    """
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


# TF-IDF类
class TfIdf(object):
    def __init__(self, idf_idc, default_idf, word_list, keyword_num):
        """
        四个参数分别是：训练好的idf字典，默认idf值，处理后的待提取文本，关键词数量
        :param idf_idc: 训练好的idf字典
        :param default_idf: 默认idf值
        :param word_list: 处理后的待提取文本
        :param keyword_num: 关键词数量
        """
        self.word_list = word_list
        self.idf_idc = idf_idc
        self.default_idf = default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num

    # 统计tf值
    def get_tf_dic(self):
        tf_idc = {}
        for word in self.word_list:
            tf_idc[word] = tf_idc.get(word, 0.0) + 1.0

        word_count = len(self.word_list)
        for k, v in tf_idc.items():
            tf_idc[k] = float(v) / word_count

        return tf_idc

    # 按公式计算tf-idf
    def get_tfidf(self):
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_idc.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)

            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        tfidf_dic.items()

        # 根据tf-idf排序，去排名前keyword_num的词作为关键词
        result_dic = sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)
        save_data = []
        for k, v in result_dic[:self.keyword_num]:
            print(k + "/ ", end='')
            # print(k + " - " + str(v) + "/ ", end='')
            save_data.append("{} - {}\n".format(k, v))
        save_to_text("../../data/txt/tfidf.txt", "".join(save_data))
        print()


# 主题模型
class TopicModel(object):
    def __init__(self, doc_list, keyword_num, model='LSI', num_topics=4):
        """
        三个传入参数：处理后的数据集，关键词数量，具体模型（LSI、LDA），主题数量
        :param doc_list:  处理后的数据集
        :param keyword_num:  关键词数量
        :param model:  具体模型（LSI、LDA）
        :param num_topics: 主题数量
        """
        # 使用gensim的接口，将文本转为向量化表示
        # 先构建词空间
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型向量化
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]

        self.keyword_num = keyword_num
        self.num_topics = num_topics
        # 选择加载的模型
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()

        # 得到数据集的主题-词分布
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lda

    def word_dictionary(self, doc_list):
        """
         词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
        :param doc_list:
        :return:
        """
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)

        dictionary = list(set(dictionary))
        return dictionary

    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}

        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        # 余弦相似度计算
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k,v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v,senttopic)
            sim_dic[k] = sim

        result_dic = sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)
        self.save_extract_data(result_dic)

        def doc2bowvec(self,word_list):
            vec_list = [1 if word in word_list else 0 for word in self.dictionary]
            return vec_list

    def save_extract_data(self, result_dic):
        save_data = []
        for k, v in result_dic[:self.keyword_num]:
            print(k + "/ ", end='')
            # print(k + " - " + str(v) + "/ ", end='')
            save_data.append("{} - {}\n".format(k, v))
        save_to_text("../../data/txt/{}.txt".format(self.model), "".join(save_data))
        print()


def tfidf_extract(word_list, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    idf_idc, default_idf = train_idf(doc_list)
    tfidf_model = TfIdf(idf_idc, default_idf, word_list, keyword_num)
    tfidf_model.get_tfidf()

def textrank_extract(text,pos=False,keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text,keyword_num)
    # 输出抽取出的关键词
    for k in keywords:
        print(k + "/ ", end='')
    print()

def topic_extract(word_list,model, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    topic_model = TopicModel(doc_list , keyword_num,model=model)
    topic_model.get_simword(word_list)

if __name__ == '__main__':
    text = '6月19日,《2012年度“中国爱心城市”公益活动新闻发布会》在京举行。' + \
           '中华社会救助基金会理事长许嘉璐到会讲话。基金会高级顾问朱发忠,全国老龄' + \
           '办副主任朱勇,民政部社会救助司助理巡视员周萍,中华社会救助基金会副理事长耿志远,' + \
           '重庆市民政局巡视员谭明政。晋江市人大常委会主任陈健倩,以及10余个省、市、自治区民政局' + \
           '领导及四十多家媒体参加了发布会。中华社会救助基金会秘书长时正新介绍本年度“中国爱心城' + \
           '市”公益活动将以“爱心城市宣传、孤老关爱救助项目及第二届中国爱心城市大会”为主要内容,重庆市' + \
           '、呼和浩特市、长沙市、太原市、蚌埠市、南昌市、汕头市、沧州市、晋江市及遵化市将会积极参加' + \
           '这一公益活动。中国雅虎副总编张银生和凤凰网城市频道总监赵耀分别以各自媒体优势介绍了活动' + \
           '的宣传方案。会上,中华社会救助基金会与“第二届中国爱心城市大会”承办方晋江市签约,许嘉璐理' + \
           '事长接受晋江市参与“百万孤老关爱行动”向国家重点扶贫地区捐赠的价值400万元的款物。晋江市人大' + \
           '常委会主任陈健倩介绍了大会的筹备情况。'

    pos = False
    seg_list = seg_to_list(text, pos)
    filter_list = word_filter(seg_list, pos)

    print('TF-IDF模型结果：')
    tfidf_extract(filter_list)
    print('TextRank模型结果：')
    textrank_extract(text)
    print('LSI模型结果：')
    topic_extract(filter_list, 'LSI', pos)
    print('LDA模型结果：')
    topic_extract(filter_list, 'LDA', pos)