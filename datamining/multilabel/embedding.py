#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : embedding.py
# @Author: sl
# @Date  : 2021/10/22 - 下午10:07

"""
fasttext 构建词向量

"""

# -*- coding: utf-8 -*-
import jieba
import os
import fasttext
import pandas as pd

from util.constant import QA_ALL_SORT_DATA_DIR, QA_DATA_DIR, QA_ALL_SORT_CUT_DATA_DIR
from util.nlp_utils import stop_words

stopword = set(stop_words())


def cut_data(text):
    seg_text = jieba.lcut(text)
    seg_text = [w for w in seg_text if w not in stopword]
    outline = " ".join(seg_text)
    return outline


def filter_data(text):
    texts = text.split()
    res = []
    for a in texts:
        a = str(a).replace("%", "").replace(".", "")
        if str(a).isnumeric() or str(a).isascii():
            continue
        res.append(a)
    return " ".join(res)


def load_data():
    df = pd.read_csv(QA_ALL_SORT_DATA_DIR, names=["gid", "question"])
    print(df.info())

    df["cut"] = df["question"].apply(cut_data)

    df.to_csv(QA_ALL_SORT_CUT_DATA_DIR, columns=["cut"], header=False, index=False)


def process_data():
    df = pd.read_csv(QA_ALL_SORT_CUT_DATA_DIR, names=["index", "question"])

    df["cut"] = df["question"].apply(filter_data)

    df.to_csv(QA_ALL_SORT_CUT_DATA_DIR, columns=["cut"], header=False, index=False)


def get_data():
    # 清华大学的新闻分类文本数据集下载：https://thunlp.oss-cn-qingdao.aliyuncs.com/THUCNews.zip
    data_dir = 'D:\\迅雷下载\\THUCNews\\THUCNews\\财经'

    with open("finance_news_cut.txt", "w", encoding='utf-8') as f:
        for file_name in os.listdir(data_dir):
            print(file_name)
            file_path = data_dir + os.sep + file_name
            with open(file_path, 'r', encoding='utf-8') as fr:
                text = fr.read()
                seg_text = jieba.cut(text.replace("\t", " ").replace("\n", " "))
                outline = " ".join(seg_text)
                f.write(outline)
                f.flush()


def train_model():
    model = fasttext.train_unsupervised(QA_ALL_SORT_CUT_DATA_DIR,  epoch=50, wordNgrams=2,dim=50)
    model.save_model("news_fasttext.model.bin")


def model_test():
    model = fasttext.load_model('news_fasttext.model.bin')
    print(len(model.words))
    print(model.words)

    print(model.get_word_vector("股票"))
    print(model.get_nearest_neighbors('股票'))


if __name__ == "__main__":
    pass
    # load_data()
    # process_data()
    res = filter_data("1992 34.3 444% 444.1%  你好 ffff fff111")
    print(res)

    # get_data()
    train_model()
    model_test()
