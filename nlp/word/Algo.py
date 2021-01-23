#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : Algo.py
# @Author: sl
# @Date  : 2021/1/12 -  下午11:26
import numpy as np

def show_row_column( data):
    res_data = np.array([data]).T
    print(res_data)
    return res_data

def remove_same_prefix( data):
    res_list = list(set(data))
    res_list.sort(key=lambda x: len(x), reverse=True)

    res = []
    before = ""
    for index, word in enumerate(res_list):
        if index == 0:
            before = word
            res.append(before)
        else:
            flag = False
            for one_word in res:
                if one_word.startswith(word):
                    flag = True
                    break
            if not flag:
                res.append(word)

    return res

def remove_input(data,remove_data):
    remove_set = set(remove_data)
    res_list = list(data)
    res = []
    for word in res_list:
        if word not in remove_set:
                res.append(word)
    return  res

if __name__ == '__main__':

    data_set = set()
    data_list = []
    data_list.append('corded telephones')
    data_list.append('telephone headsets')
    data_list.append('telephones')
    data_list.append('telephone')
    data_list.append('mobile telephone hello world')
    data_list.append('mobile telephone')
    data_list.append('mobile telephone word')
    data_list.append('mobile telephone hello')
    data_list.append('mobile ')


    # res_list = list(set(data_list))
    # print(res_list)
    #
    # res_list2 = show_row_column(res_list)
    #
    # a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    #
    # # res = map(list, zip(*a))
    # res = np.transpose(a)
    # res = np.transpose(data_list)
    # print(res)

    # res = remove_same_prefix(data_list)
    r_list = ['telephones','corded telephones']
    res = remove_input(data_list,r_list)
    print(res)

    pass