#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_fp_growth_first.py
# @Author: sl
# @Date  : 2020/10/25 - 上午11:47

import datamining.association_rules.fp_growth_first as fpg

# 数据集
dataset = [
    ['啤酒', '牛奶', '可乐'],
    ['尿不湿', '啤酒', '牛奶', '橙汁'],
    ['啤酒', '尿不湿'],
    ['啤酒', '可乐', '尿不湿'],
    ['啤酒', '牛奶', '可乐']
]


if __name__ == '__main__':

    '''
    调用find_frequent_itemsets()生成频繁项
    @:param minimum_support表示设置的最小支持度，即若支持度大于等于inimum_support，保存此频繁项，否则删除
    @:param include_support表示返回结果是否包含支持度，若include_support=True，返回结果中包含itemset和support，否则只返回itemset
    '''
    min_support = 0.2
    min_confidence = 0.8
    data_length = len(dataset)
    minimum_support = min_support * data_length
    print("支持度大小: {} -> 总的数据量：{} ->设置的支持度大小：{}".format(minimum_support, data_length, min_support))
    frequent_itemsets = fpg.find_frequent_itemsets(dataset, minimum_support=minimum_support, include_support=True)
    # print(type(frequent_itemsets))   # print type

    result = []
    frequentPatterns = {}
    for itemset, support in frequent_itemsets:    # 将generator结果存入list
        result.append((itemset, support))
        frequentPatterns[frozenset(itemset)] = support
    result = sorted(result, key=lambda i: i[0])   # 排序后输出

    print("-------------------------- 频繁项集 --------------------------")
    for itemset, support in result:
        print("{} - {} - {} ".format(str(itemset), str(support), support / data_length ))

    minConf = 0.8
    rules = []
    fpg.rules_generator(frequentPatterns, minConf, rules)

    print()
    print("-------------------------- 关联规则 --------------------------")

    for (from_items, to_items, confidients) in rules:
        print("{} -> {} -> {}".format(set(from_items), set(to_items), confidients))