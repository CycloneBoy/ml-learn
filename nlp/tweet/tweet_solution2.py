#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : tweet_solution.py
# @Author: sl
# @Date  : 2020/10/14 - 下午8:58

import csv
import re
import os

import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from util.constant import WORK_DIR

pd.set_option('display.max_colwidth', -1)

data_dir = "{}/data/txt".format(WORK_DIR)

stop = stopwords.words('english')
print("stopwords size:{}".format(len(stop)))

word_index = "Main_Narrative"
filename = "{}/{}".format(data_dir, 'police.csv')
save_filename = "{}/{}".format(data_dir, 'police_clean.csv')


def test_print_csv(filename):
    res = []
    with open(filename, encoding="unicode_escape") as f:
        reader = csv.reader(f)
        # print(list(reader))
        for row in reader:
            # print(row)
            res.append(row)
    return res


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


def transfrom_dic(data):
    pass


# 文本常常包含许多特殊字符，这些字符对于机器学习算法来说不一定有意义。因此，我要采取的第一步是删除这些
def clean_text(data):
    header = data[0]
    data = data[1:]
    result = []
    result.append(header)
    for row in data:
        line = str(row[1]).lower()
        res = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]', "", line)
        onerow = []
        onerow.append(row[0])
        onerow.append(res)
        result.append(onerow)
    return result


def get_stopset(stopwords):
    stop_set = set()
    for word in stopwords:
        stop_set.add(word)
    return stop_set


def clean_stopword(line, stop_set):
    split = line.split(" ")
    res = []
    for word in split:
        if word not in stop_set and len(word) > 0:
            res.append(word)
    return res


def remove_stopwords(data, stopwords):
    result = []
    stop_set = get_stopset(stopwords)
    for row in data:
        line = str(row[1]).lower()
        res = clean_stopword(line, stop_set)
        res = test_english(res)
        if len(res) > 0:
            onerow = []
            onerow.append(row[0])
            onerow.append(" ".join(res))
            result.append(onerow)
    return result


def test_english(data):
    res = []
    for word in data:
        if wordnet.synsets(word):
            res.append(word)
    return res


def print_txt(result):
    for str in result:
        print(str)


def save_csv(filename,data):
    with open(filename, mode='w', encoding='utf-8') as f:
        for word in data:
            f.write(",".join(word))
            f.write(os.linesep)


def process_data():
    res = test_print_csv(filename)
    print("before size %s" % len(res))
    result = parse_data(res)
    print("after size %s" % len(result))
    result = clean_text(result)
    print("after clean_text size %s" % len(result))
    # print_txt(result)
    result = remove_stopwords(result, stop)
    print("after remove_stopwords size %s" % len(result))
    save_csv(save_filename, result)

def print_text(data):
    print(data.head())



if __name__ == '__main__':
    # process_data()

    # print_txt(result)

    sample_submission = pd.read_csv(save_filename)
    print_text(sample_submission)

    pass
