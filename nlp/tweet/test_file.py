#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_file.py
# @Author: sl
# @Date  : 2020/10/15 - 下午9:50
import os
import re

from nltk.corpus import stopwords
from nltk.corpus import wordnet

from util.constant import WORK_DIR

stop = stopwords.words('english')

word_index = "Main_Narrative"

data_dir = "{}/data/txt".format(WORK_DIR)

# filename = "{}/{}".format(data_dir, 'test1.csv')
filename = "{}/{}".format(data_dir, 'police1.csv')
save_filename = "{}/{}".format(data_dir, 'police_parse.csv')


def write_file():
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write("test1")
        f.write(os.linesep)
        f.write("test2")


def clean_word():
    line = "1cad job refers  2 victim and poi  ex partners  4 dvi recidivist  5 three children  6 the victim and "
    res = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]', "", line)
    print(res)


def clean_stopword(line, stopwords):
    stop_set = set()
    for word in stopwords:
        stop_set.add(word)
    split = line.split(" ")
    # print(split)
    res = []
    for word in split:
        if word not in stop_set and len(word) > 0:
            res.append(word)
    return res


def test_english(data):
    res = []
    for word in data:
        if wordnet.synsets(word):
            res.append(word)
    return res


if __name__ == '__main__':
    # write_file()

    # clean_word()

    line = "i me he test we our youself cad number counter complaint   parties involved ex partners   months  recorded dvi¡¯s  previous this is th dvi   children  nil children between the two partes    account  on th december  poi was served with vro protecting victim from him   victim has been advised to report any unusual  suspcious behaviour due to the ongoing issues with her ex partne";
    res = clean_stopword(line, stop)
    print(" ".join(res))

    res = test_english(res)
    print(" ".join(res))

    pass
