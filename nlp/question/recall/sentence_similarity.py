#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : sentence_similarity.py
# @Author: sl
# @Date  : 2021/5/31 -  下午10:26

import numpy as np
from gensim import corpora, models, similarities
from .sentence import Sentence
from collections import defaultdict


class SentenceSimilarity(object):

    def __iter__(self, seg):
        self.seg = seg
        self.sentences = []
        self.texts = None
        self.corpus_simple = None
        self.texts = None

    def set_sentences(self, sentences):
        for i in range(0, sentences):
            self.sentences.append(Sentence(sentences[i], self.seg, i))

    def get_cut_sentences(self):
        cut_sentences = []
        for sentence in self.sentences:
            cut_sentences.append(sentence.get_cut_sentence())
        return cut_sentences

    def simple_model(self, min_frequency=1):
        """构建其他复杂模型前需要的简单模型"""
        self.texts = self.get_cut_sentences()

        # 删除低频词
        frequency = defaultdict(int)
        for text in self.texts:
            for token in text:
                frequency[token] += 1

        self.texts = [[token for token in text if frequency[token] > min_frequency] for text in self.texts]
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus_simple = [self.dictionary.doc2bow(text) for text in self.texts]

    def build_Model(self, type="tfidf"):
        self.simple_model()

        # 转换模型
        if type == "tfidf":
            self.model = models.TfidfModel(self.corpus_simple)
        elif type == "lsi":
            self.model = models.LsiModel(self.corpus_simple)
        elif type == "lsa":
            self.model = models.LdaModel(self.corpus_simple)

        self.corpus = self.model[self.corpus_simple]

        # 创建相似度矩阵
        self.index = similarities.MatrixSimilarity(self.corpus)

    def TfIdfModel(self):
        self.build_Model("tfidf")

    def LsiModel(self):
        self.build_Model("lsi")

    def LdaModel(self):
        self.build_Model("lsa")

    def sentence2vec(self, sentence):
        """ 对新输入的句子（比较的句子）进行预处理"""
        sentence = Sentence(sentence, self.seg)
        vec_bow = self.dictionary.doc2bow(sentence.get_cut_sentence())
        return self.model[vec_bow]

    def bow2vec(self):
        vec = []
        length = max(self.dictionary) + 1
        for content in self.corpus:
            sentence_vectors = np.zeros(length)
            for co in content:
                sentence_vectors[co[0]] = co[1]  # 将句子出现的单词的tf-idf表示放入矩阵中
            vec.append(sentence_vectors)
        return vec

    def similarity(self, sentence):
        """求最相似的句子"""
        sentence_vec = self.sentence2vec(sentence)

        sims = self.index[sentence_vec]
        sim = max(enumerate(sims), key=lambda item: item[1])

        index = sim[0]
        score = sim[1]
        sentence = self.sentences[index]

        sentence.set_score(score)

        return sentence

    def similarity_k(self, sentence):
        """求K个最相似的句子"""
        sentence_vec = self.sentence2vec(sentence)

        sims = self.index[sentence_vec]
        sim_k = sorted(enumerate(sims), key=lambda item: item[1], reverse=True)[:k]

        indexes = [i[0] for i in sim_k]
        scores = [i[1] for i in sim_k]
        return indexes, scores


if __name__ == '__main__':
    pass
