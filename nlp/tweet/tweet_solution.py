#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : tweet_solution.py
# @Author: sl
# @Date  : 2020/10/14 - 下午8:58

import re
import csv

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from util.constant import WORK_DIR

pd.set_option('display.max_colwidth', -1)

data_dir = "{}/data/txt".format(WORK_DIR)

# nltk.download('stopwords')

stop = stopwords.words('english')
print("stopwords size:{}".format(len(stop)))


def print_text(data):
    print(data.head())


word_index = "Main_Narrative"
filename = "{}/{}".format(data_dir, 'police_parse.csv')

test_data = pd.read_csv(filename,encoding="unicode_escape")
# test_data = pd.read_csv("{}/{}".format(data_dir, 'police.csv'),encoding='gbk')
test_data.head()

print_text(test_data)
print_text(test_data[word_index])

print("size: {}".format(len(test_data[word_index])))

def parse_data(list):
    result = []
    index = 1
    last_row = None
    one_row = ""
    while index < len(list):
        row_len = len(list[index][0])
        if row_len == 0:
            # print(row)
            one_row += " ".join([x for x in list[index]])
            pass
        else:
            if last_row is not None:
                last_row[2] = last_row[2] + one_row
                one_row = ""
                result.append(last_row)
            last_row = list[index]
        index += 1
    return result

def read_csv(filename):
    res = []
    with open(filename,encoding="unicode_escape") as f:
        reader = csv.reader(f)
        # print(list(reader))
        for row in reader:
            # print(row)
            res.append(row)
    return res


# 文本常常包含许多特殊字符，这些字符对于机器学习算法来说不一定有意义。因此，我要采取的第一步是删除这些
def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(
        lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    return df


data_clean = clean_text(test_data, word_index)
print_text(data_clean)


data_clean[word_index] = data_clean[word_index].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))
print_text(data_clean[word_index])

if __name__ == '__main__':
    pass