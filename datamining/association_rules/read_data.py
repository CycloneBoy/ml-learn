#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : read_data.py
# @Author: sl
# @Date  : 2020/10/25 - 下午12:03
import csv


# 读取数据
def read_data(file_name):
    dataset = []
    with open(file_name, 'r', encoding="utf-8") as f:
        for cell in csv.reader(f):
            row = []
            for item in cell:
                row.append(str(item))
            dataset.append(row)
    header = dataset[0]
    return header,dataset[1:]

# 处理表头
def transfrom_header(header):
    header_dic = {}
    for index, var in enumerate(header):
        # print("[{}] = {}".format(index, var))
        header_dic[str(index)] = var
    return header_dic


# 处理　accident_type_set
def parse_accident_type(dataset,header_dic,type_index=3):
    # print("total:{}".format(len(dataset)))
    accident_type_set = {}
    for idx, row in enumerate(dataset):
        accident_type = None
        data = []
        for index, cell in enumerate(row):
            if index == type_index:
                accident_type = cell
            if ((8 <= index <= 14) or (19 <= index <= 29)) and str(cell).strip() == "1":
                data.append(header_dic[str(index)])

        # print("idx:{} -> {}".format(idx,row))
        # print("idx:{} -> {}".format(idx,str(data)))

        data_list = []
        try:
            keys = set(accident_type_set)
            if accident_type in keys:
                data_list = accident_type_set[accident_type]
            data_list.append(data)
            accident_type_set[accident_type] = data_list
        except KeyError:
            print("error in KeyedVectors")
    key_list = []
    for key in accident_type_set.keys():
        key_list.append(key)
    return key_list, accident_type_set


# 打印　accident_type_set
def print_accident_type_set(accident_type_keys,accident_type_set):
    print("keys数量:{}".format(str(accident_type_keys)))
    for key in accident_type_keys:
        print("keys :{} -> 总量：{}".format(key,len( accident_type_set.get(key))))

    print("Keys总量-> len:{}".format(len(accident_type_set)))
    total = 0
    for keys in accident_type_set.keys():
        var = accident_type_set.get(keys)
        print("keys:{} ->{}".format(keys, len(var)))
        total += len(var)
        for index, row in enumerate(var):
            print("{} -> {}".format(index, row))
        print("--------------------------------------")
    print("total:{}".format(total))


def read_data_by_type(data_file,type_index = 3):
    header, dataset = read_data(data_file)
    header_dic = transfrom_header(header)

    accident_type_keys, accident_type_set = parse_accident_type(dataset, header_dic,
                                                                type_index=type_index)
    # print_accident_type_set(accident_type_keys,accident_type_set)

    return accident_type_keys, accident_type_set

if __name__ == '__main__':

    data_file = '/home/sl/workspace/python/a2020/ml-learn/data/test/analysis_1025.csv'
    accident_type_index = 3
    accident_consequence_index = 4

    # header,dataset = read_data(data_file)
    # header_dic = transfrom_header(header)
    # # accident_type_keys,accident_type_set = parse_accident_type(dataset,header_dic,type_index=accident_type_index)
    # accident_type_keys,accident_type_set = parse_accident_type(dataset,header_dic,type_index=accident_consequence_index)
    # print_accident_type_set(accident_type_set)
    read_data_by_type(data_file,accident_type_index)

