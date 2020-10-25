#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : read_data.py
# @Author: sl
# @Date  : 2020/10/12 - 下午10:28

# import nltk

# nltk.download()

# nltk.download('stopwords')


# import nltk
# from nltk.book import *

# import nltk
#
# sen = 'hello, how are you?'
# res = nltk.word_tokenize(sen)
# print(res)

import pandas as pd

pd.set_option('display.max_colwidth', -1)

import csv
import os

from util.constant import WORK_DIR

word_index = "Main_Narrative"

data_dir = "{}/data/txt".format(WORK_DIR)

# filename = "{}/{}".format(data_dir, 'test1.csv')
filename = "{}/{}".format(data_dir, 'police.csv')
save_filename = "{}/{}".format(data_dir, 'police_parse.csv')



def test_print_csv():
    res = []
    with open(filename,encoding="unicode_escape") as f:
        reader = csv.reader(f)
        # print(list(reader))
        for row in reader:
            # print(row)
            res.append(row)
    return res


def test_pd():
    df = pd.read_csv(filename, encoding="unicode_escape")
    print(df.head(10))
    print(df.tail())






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

def save_csv(data):
    with open(save_filename, 'w',encoding="unicode_escape") as f:
        writer = csv.writer(f)
        header = []
        header.append("IR_No")
        header.append("Main_Narrative")
        header.append("FVIR_FVNarrative")
        f.write(",".join(header))
        f.write(os.linesep)
        # writer.writerow(header)
        for row in data:
            header = []
            header.append(row[0])
            header.append(row[1])
            header.append(row[2])
            f.write(",".join(header))
            f.write(os.linesep)
            # writer.writerow(header)


def save_csv2(data,filename):
    with open(filename,mode='w',encoding='utf-8') as f:
        f.write("IR_No,Main_Narrative")
        f.write(os.linesep)
        for str in data:
            line = str[0] +"," + str[1] + str[2]
            f.write(line)
            f.write(os.linesep)


if __name__ == '__main__':
    res = test_print_csv()
    print("before size %s" % len(res))


    # for idx,row in enumerate(res[1:]):
    #     # print("{} - {}".format(idx,row))
    #     if len(row[0]) == 0:
    #         # print(row)
    #         pass
    #     else:
    #         result.append(row)

    result = parse_data(res)
    print("after size %s" % len(result))

    # for str in result:
    #     print(str)

    # save_csv(result)
    save_csv2(result,save_filename)


    # with open(save_filename,mode='w',encoding='utf-8') as f:
    #     for str in result:
    #         f.write(",".join(str)[:-1])
    #         f.write(os.linesep)

    # test_pd()
    pass
