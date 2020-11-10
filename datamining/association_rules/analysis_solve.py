#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : analysis_solve.py
# @Author: sl
# @Date  : 2020/10/25 - 下午9:23
import os

import datamining.association_rules.fp_growth_first as fpg
import datamining.association_rules.read_data as read_data


from pyecharts import options as opts
from pyecharts.charts import Graph

colors = [
"#c71934",
"#84c719",
"#c71984",
"#c71919",
"#8419c7",
"#1919c7",
"#19c7b9",
"#194fc7",
"#c7b919",
"#69c719",
"#6919c7",
"#9f19c7",
"#b9c719",
"#19c734",
"#3419c7",
"#c71969",
"#c76919",
"#1984c7",
"#34c719",
"#c78419",
"#1969c7",
"#b919c7",
"#19b9c7",
"#4fc719",
"#19c74f",
"#199fc7",
"#c719b9",
"#c7199f",
"#19c784",
"#c74f19",
"#9fc719",
"#19c79f",
"#c73419",
"#4f19c7",
"#c79f19",
"#19c719",
"#19c769",
"#1934c7",
"#c7194f"
]


def get_node(frequent_itemsets):
    node_set = set()
    for itemset, support in frequent_itemsets:  # 将generator结果存入list
        for item in itemset:
            node_set.add(item )

    all_node = {}
    all_link = []

    for node in node_set:
        all_node[node] = 0

    for itemset, support in frequent_itemsets:
        for item in itemset:
            all_node[item] = all_node[item] + support
        #　边
        if len(itemset) >=2:
            first = itemset[0]
            for idx ,item in enumerate(itemset[1:]):
                all_link.append((first,item,support))
                first = item


    print("-------------------------- 节点数据 --------------------------")
    print("节点数据总数：{}".format(len(all_node)))

    # 所有的node
    for key in all_node.keys():
        print("{} - {}".format(key,all_node[key]))


    print("-------------------------- 边数据 --------------------------")
    print("节边数据总数：{}".format(len(all_link)))

    # 所有的node
    for first,item,support in all_link:
        print("{} -> {} : {}".format(first,item,support))

    return all_node,all_link

# 打印频繁项集
def print_frequent_patterns(frequent_itemsets):
    result = []
    frequentPatterns = {}

    for itemset, support in frequent_itemsets:  # 将generator结果存入list
        result.append((itemset, support))
        frequentPatterns[frozenset(itemset)] = support
    result = sorted(result, key=lambda i: i[0])  # 排序后输出
    print("-------------------------- 频繁项集 --------------------------")
    print("频繁项集总数：{}".format(len(result)))

    for itemset, support in result:
        print("{} - {} - {} ".format(str(itemset), str(support), support / data_length))

    return result,frequentPatterns

# 打印关联规则
def print_rules(rules):
    print()
    print("-------------------------- 关联规则 --------------------------")
    print("关联规则总数：{}".format(len(rules)))
    for (from_items, to_items, confidients) in rules:
        print("{} -> {} -> {}".format(set(from_items), set(to_items), confidients))

def save_data(rules,file_name):
    with open(file_name, 'w', encoding="utf-8") as f:
        for (from_items, to_items, confidients) in rules:
            # print("{} -> {} -> {}".format(set(from_items), set(to_items), confidients))
            line = "{} -> {} -> {}".format(set(from_items), set(to_items), confidients)
            f.write(line)
            f.write(os.linesep)
    print("保存结果")

if __name__ == '__main__':

    data_file = '/home/sl/workspace/python/a2020/ml-learn/data/test/analysis_1025.csv'
    accident_type_index = 3
    accident_consequence_index = 4
    min_support = 0.02
    min_confidence = 0.8

    accident_type_keys, accident_type_set = read_data.read_data_by_type(data_file, accident_consequence_index)

    print("类型大小：{}".format(len(accident_type_keys)))

    dataset = accident_type_set[accident_type_keys[0]]
    data_length = len(dataset)
    minimum_support = min_support * data_length
    print("支持度大小: {} -> 总的数据量：{} ->设置的支持度大小：{}".format(minimum_support, data_length, min_support))

    frequent_itemsets = fpg.find_frequent_itemsets(dataset, minimum_support=minimum_support, include_support=True)


    result,frequentPatterns = print_frequent_patterns(frequent_itemsets)
    all_node,all_link = get_node(result)


    rules = []
    fpg.rules_generator(frequentPatterns, min_confidence, rules)

    print_rules(rules)
    save_data(rules,'consequence_rules_002.txt')

    nodes = []
    links = []
    nodes.append(opts.GraphNode(name=accident_type_keys[0], symbol_size=len(dataset)/4))
    for key in all_node.keys():
        nodes.append(opts.GraphNode(name=key,
                                    symbol_size=all_node[key]/100,
                                    label_opts={"normal": {"color": colors[12]}}))
        links.append(opts.GraphLink(source=accident_type_keys[0], target=key, value=all_node[key]/100))


    for first,item,support in all_link:
        links.append(opts.GraphLink(source=first, target=item,value=support/10,symbol_size=support/10))



    print("graph_{}.html".format(accident_type_keys[0]))
    c = (
        Graph(init_opts=opts.InitOpts(width="1000px", height="600px"))
            .add("", nodes, links, repulsion=4000,
                 is_roam=True,
                 is_focusnode=True
                 )
            .set_global_opts(title_opts=opts.TitleOpts(title="Graph-"+ accident_type_keys[0]))
            .render("graph_{}.html".format(accident_type_keys[0]))
    )




