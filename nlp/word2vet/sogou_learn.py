#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : sogou_learn.py
# @Author: sl
# @Date  : 2021/7/3 -  下午2:31

import gensim

from util.constant import DATA_EMBEDDING_SOGOU_CHAR, DATA_EMBEDDING_FINANCIAL_BIGRAM_CHAR


def atest_model(model_name=DATA_EMBEDDING_SOGOU_CHAR):
    print("开始加载词向量完毕")
    model = gensim.models.KeyedVectors.load_word2vec_format(model_name,  binary=False, unicode_errors='ignore')
    # print(help(model.most_similar))
    print("加载词向量完毕")
    res = model.most_similar('北京大学')
    print(res)

    print(len(model.wv['北京大学']))
    print(model.wv['北京大学'])

    testwords = ['金融', '股票', '经济']
    for w in testwords:
        print('*' * 100)
        print(w)
        print(model.most_similar(w))


    pass

if __name__ == '__main__':
    # atest_model()
    atest_model(model_name=DATA_EMBEDDING_FINANCIAL_BIGRAM_CHAR)
    pass