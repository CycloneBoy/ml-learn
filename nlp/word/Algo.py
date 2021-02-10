#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : Algo.py
# @Author: sl
# @Date  : 2021/1/12 -  下午11:26
import numpy as np


def show_row_column(data):
    res_data = np.array([data]).T
    print(res_data)
    return res_data


def remove_same_prefix(data):
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


def remove_input(data, remove_data):
    remove_set = set(remove_data)
    res_list = list(data)
    res = []
    for word in res_list:
        if word not in remove_set:
            res.append(word)
    return res


# sort reverse
def show_list_right(data):
    max_len = 0
    for item in data:
        if len(item) > max_len:
            max_len = len(item)

    sort_list = []
    for item in data:
        sort_list.append(str(item).rjust(max_len))

    return sort_list


# sort reverse
def sort_list_right(data):
    sort_list = []
    # 反转字符串
    for item in data:
        sort_list.append(''.join(reversed(item)))

    # 排序
    sort_list.sort(key=lambda x: str(x).lower())

    res_list = []
    # 反转字符串
    for item in sort_list:
        res_list.append(''.join(reversed(item)))

    return res_list


# 显示字符串列表
def show_list(data, show_right=False):
    sort_list = []
    if show_right:
        sort_list = show_list_right(data)
    else:
        sort_list = data

    print("---------------------------")
    for item in sort_list:
        print(item)
    return sort_list

# 去除相同后缀的
def remove_same_suffix(data, remove_data):
    remove_set = get_suffix_list(remove_data)
    res_list = list(data)
    res = []
    for word in res_list:
        add_flag = True
        split_list = str(word).split()
        word = " ".join(split_list)
        for remove_item in remove_set:
            if str(word).strip().find(remove_item) != -1:
                add_flag = False
                break
        if add_flag:
            res.append(word)
    return res

# 生成后缀列表
def get_suffix_list(data):
    res = []
    for item in data:
        split_list = str(item).split()
        # print(split_list)
        while len(split_list) >= 1:
            res.append(" ".join(split_list))
            split_list = split_list[1:]

    return set(res)

if __name__ == '__main__':
    data_set = set()
    data_list = []
    # data_list.append('corded telephones')
    # data_list.append('telephone headsets')
    # data_list.append('telephones')
    # data_list.append('telephone')
    # data_list.append('mobile telephone hello world')
    # data_list.append('mobile telephone')
    # data_list.append('mobile telephone word')
    # data_list.append('mobile telephone hello')
    # data_list.append('mobile ')
    data_list.append('cell phone mount')
    data_list.append('d cell phone mount')
    data_list.append('e cell  phone mount')
    data_list.append('g cell   phone mount')
    data_list.append('a cell phone    mount')
    data_list.append('c cell phone mount')
    data_list.append('b cell phone mount')
    data_list.append('f cell     phone mount')
    data_list.append('universal cell phone mount')
    data_list.append('clip cell phone mount')
    data_list.append('cup holder cell phone mount')
    data_list.append('motocycle cell phone mount')
    data_list.append('site short cell phone mount')
    data_list.append('mactream camera tripod with phone mount')
    data_list.append('selfie led ring ligth with phone mount')
    data_list.append('test with phone')
    data_list.append('test with mount')

    # res = remove_same_prefix(data_list)
    # r_list = ['cell phone mount', 'with phone']
    r_list = ['mount']
    # res = show_list_right(data_list)
    res = sort_list_right(data_list)
    show_list(res, True)

    res2 = remove_same_suffix(res,r_list)
    show_list(res2, True)

    print("----------------")
    #
    # r_list = ['test this is a cell phone mount', 'corded telephones phone mount','phone mount']
    # res2 = get_suffix_list(r_list)
    # show_list(res2, True)
    # pass
