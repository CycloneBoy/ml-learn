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

def removeStr(set, str):
    tempSet = []
    for elem in set:
        if(elem != str):
            tempSet.append(elem)
    tempFrozenSet = frozenset(tempSet)
    return tempFrozenSet

def rulesGenerator(frequentPatterns, minConf, rules):
    for frequentset in frequentPatterns:
        if(len(frequentset) > 1):
            getRules(frequentset,frequentset, rules, frequentPatterns, minConf)

def getRules(frequentset,currentset, rules, frequentPatterns, minConf):
    for frequentElem in currentset:
        subSet = removeStr(currentset, frequentElem)
        confidence = frequentPatterns[frequentset] / frequentPatterns[subSet]
        if (confidence >= minConf):
            flag = False
            for rule in rules:
                if(rule[0] == subSet and rule[1] == frequentset - subSet):
                    flag = True
            if(flag == False):
                rules.append((subSet, frequentset - subSet, confidence))

            if(len(subSet) >= 2):
                getRules(frequentset, subSet, rules, frequentPatterns, minConf)

def calc_conf(freqSet, H, supportData, brl, minConf=0.7):
    ''' 计算可信度（confidence）

    :param freqSet: 频繁项集中的元素，例如: frozenset([1, 3])
    :param H: 频繁项集中的元素的集合，例如: [frozenset([1]), frozenset([3])]
    :param supportData:  所有元素的支持度的字典
    :param brl: 关联规则列表的空数组
    :param minConf: 最小可信度
    :return:  记录 可信度大于阈值的集合
    '''
    # 记录可信度大于最小可信度（minConf）的集合
    prunedH = []
    for conseq in H:  # 假设 freqSet = frozenset([1, 3]), H = [frozenset([1]), frozenset([3])]，那么现在需要求出 frozenset([1]) -> frozenset([3]) 的可信度和 frozenset([3]) -> frozenset([1]) 的可信度
        conf = supportData[freqSet] / supportData[
            freqSet - conseq]  # 支持度定义: a -> b = support(a | b) / support(a). 假设  freqSet = frozenset([1, 3]), conseq = [frozenset([1])]，那么 frozenset([1]) 至 frozenset([3]) 的可信度为 = support(a | b) / support(a) = supportData[freqSet]/supportData[freqSet-conseq] = supportData[frozenset([1, 3])] / supportData[frozenset([1])]
        if conf >= minConf:
            # 只要买了 freqSet-conseq 集合，一定会买 conseq 集合（freqSet-conseq 集合和 conseq集合 是全集）
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

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
    print(type(frequent_itemsets))   # print type

    result = []
    frequentPatterns = {}
    for itemset, support in frequent_itemsets:    # 将generator结果存入list
        result.append((itemset, support))
        frequentPatterns[frozenset(itemset)] = support
    result = sorted(result, key=lambda i: i[0])   # 排序后输出

    for itemset, support in result:
        print("{} - {} - {} ".format(str(itemset), str(support), support / data_length ))

    minConf = 0.8
    rules = []
    rulesGenerator(frequentPatterns, minConf, rules)

    print("association rules:")

    for (from_items, to_items, confidients) in rules:
        print("{} -> {} -> {}".format(set(from_items), set(to_items), confidients))