#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : analysis_solve.py
# @Author: sl
# @Date  : 2020/10/25 - 下午9:23

import datamining.association_rules.fp_growth_first as fpg
import datamining.association_rules.read_data as read_data

import os
from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.charts import Scatter

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

type_set = set()
type_set.add("Sinking")
type_set.add("Contact")
type_set.add("Collision")
type_set.add("Fire and explosion")
type_set.add("Others")
type_set.add("Grounding")

consequence_set = set()
consequence_set.add("Ordinary")
consequence_set.add("Minor")
consequence_set.add("Serious")
consequence_set.add("Major")

visibility_set = set()
visibility_set.add("Good")
visibility_set.add("Bad")
visibility_set.add("Medium")


index = 2

# 根据关联规则绘制图
def get_node2(frequent_itemsets):
    node_set = set()
    for (from_items, to_items, confidients) in frequent_itemsets:  # 将generator结果存入list
        itemset = from_items.union(to_items)
        for item in itemset:
            node_set.add(item )

    all_node = {}
    all_link = []

    for node in node_set:
        all_node[node] = 0

    # todo
    for (from_items, to_items, confidients) in frequent_itemsets:
        itemset = from_items.union(to_items)
        for item in itemset:
            all_node[item] = all_node[item] + confidients
        #　边
        node_list = []
        for item in from_items:
            node_list.append(item)
        for item in to_items:
            node_list.append(item)

        node_list.sort()

        for idx in range(1,len(node_list)):
            all_link.append((node_list[idx -1],node_list[idx],confidients))


    print("-------------------------- 节点数据 --------------------------")
    print("节点数据总数：{}".format(len(all_node)))

    # 所有的node
    for key in all_node.keys():
        print("{} - {}".format(key,all_node[key]))


    print("-------------------------- 边数据 --------------------------")
    print("节边数据总数：{}".format(len(all_link)))
    for f1,t2,conf in all_link:
        print("{} - {} : {}".format(f1, t2,conf))

    return all_node,all_link


# 打印频繁项集
def print_frequent_patterns(frequent_itemsets,data_length):
    result = []
    frequentPatterns = {}

    for itemset, support in frequent_itemsets:  # 将generator结果存入list
        result.append((itemset, support))
        frequentPatterns[frozenset(itemset)] = support
    result = sorted(result, key=lambda i: i[0])  # 排序后输出
    print("-------------------------- 频繁项集 --------------------------")
    print("频繁项集总数：{}".format(len(result)))


    node_set = set()
    link_list = []
    for itemset, support in result:
        items = set()
        for i in itemset:
            items.add(i)
        type_res = items.intersection(type_set)
        consequence_res = items.intersection(consequence_set)

        res = []
        not_res = type_res.union(consequence_set)
        for i in itemset:
            node_set.add(i )
            if i  not in not_res :
                res.append(i)

        if len(type_res) > 0 and len(consequence_res) > 0:
            node_list_out = []
            node_list_out.append(type_res)
            node_list_out.append(consequence_res)
            for n1 in res:
                if len(n1) > 0:
                    node_list_out.append(n1)


            for idx   in range(len(node_list_out) - 1):
                link_list.append(opts.GraphLink(source=node_list_out[idx], target=node_list_out[idx+1], value=1, symbol_size=1))
            # link_list.append((type_res,consequence_res,support / data_length))
            print("{} - {} - {} - {} -{} ".format(type_res,consequence_res,str(res), str(support), support / data_length))

    return result,frequentPatterns,link_list

def process_node(frequent_itemsets,data_length):
    pass


# 打印关联规则
def print_rules_v2(rules):
    print()
    print("-------------------------- 关联规则 --------------------------")
    print("关联规则总数：{}".format(len(rules)))
    for (from_items, to_items, confidients) in rules:
        print("{} -> {} -> {}".format(set(from_items), set(to_items), confidients))



# 打印关联规则
def print_rules(rules):
    print()
    print("-------------------------- 关联规则 --------------------------")
    print("关联规则总数：{}".format(len(rules)))
    total = 0
    result = []
    for (from_items, to_items, confidients) in rules:

        item_set = set()
        for i in from_items:
            item_set.add(i)

        if len(item_set.intersection(type_set.union(consequence_set))) >= 2:
            total += 1
            print("{} -> {} -> {}".format(set(from_items), set(to_items), confidients))
            result.append((from_items, to_items, confidients))
    print("过滤后的关联规则总数：{}".format(total))
    return result

# 归一化大小
def node_normalize(all_node):
    result = {}
    node_list = []
    for key in all_node.keys():
        node_list.append(all_node[key])
    node_max = max(node_list)
    node_min = min(node_list)

    node_gap = node_max - node_min
    node_size = 0
    if node_gap == 0:
        node_gap = 1
        node_min = 0
    for key in all_node.keys():
        result[key] = (all_node[key] - node_min)/node_gap * 10
        node_size += result[key]
    return result,node_size/len(all_node.keys()) * 3



# 构造一个数据
def build_node_and_link(all_node,all_link):
    nodes = []
    links = []

    all_node,node_size = node_normalize(all_node)

    node_dic = {
        "name": "node1",
        "value": node_size,
        "symbolSize": node_size,
        "itemStyle": {
            "color": colors[index]
        }
    }
    nodes.append(node_dic)
    for key in all_node.keys():

        node_dic = {
            "name":key,
            "value": all_node[key],
            "symbolSize": all_node[key],
            "itemStyle": {
                "color": colors[index]
            }
        }
        nodes.append(node_dic)
        # nodes.append(opts.GraphNode(name=key,
        #                             symbol_size=all_node[key],
        #                             label_opts={"normal": {"color": colors[12]}}))
        # links.append(opts.GraphLink(source="node1", target=key, value=all_node[key]))
    for first, item, support in all_link:
        links.append(opts.GraphLink(source=first, target=item, value=support / 10, symbol_size=support / 10))

    return nodes,links

# 构造一个数据
def build_node_and_link_v2(all_node,all_link):
    nodes = []
    links = []

    all_node,node_size = node_normalize(all_node)

    for idex,key in enumerate(all_node.keys()):
        node_dic = {
            "name":key,
            "value": all_node[key],
            "symbolSize": all_node[key],
            "itemStyle": {
                "color": colors[idex % len(colors)]
            }
        }
        nodes.append(node_dic)

    for first, item, support in all_link:
        links.append(opts.GraphLink(source=first, target=item, value=support , symbol_size=support * 1.5))

    return nodes,links


def filter_frequent_itemsets(frequent_itemsets,data_set):
    result = []
    frequentPatterns = {}
    accident_type_set = set()
    for res in data_set["accident_type"]:
        accident_type_set.add(res)
    for itemset, support in frequent_itemsets:  # 将generator结果存入list
        itemset_set = set()
        for item in itemset:
            itemset_set.add(item)
        type_inter_set = itemset_set.intersection(accident_type_set)
        consequence_inter_set = itemset_set.intersection(data_set["accident_consequence"])
        if len(itemset) >= 4 and len(type_inter_set) >= 1 and len(consequence_inter_set) >= 1:
            result.append((itemset,support))
            frequentPatterns[frozenset(itemset)] = support
    return result,frequentPatterns

# 处理一种类别
def process_one_type(result_list,data_set):
    dataset = result_list
    data_length = len(dataset)
    minimum_support = min_support * data_length
    print("支持度大小: {} -> 总的数据量：{} ->设置的支持度大小：{}".format(minimum_support, data_length, min_support))
    frequent_itemsets = fpg.find_frequent_itemsets(dataset, minimum_support=minimum_support, include_support=True)
    # 过滤频繁项集
    # res,frequent_itemsets2 =   filter_frequent_itemsets(frequent_itemsets,data_set)

    result, frequentPatterns = print_frequent_patterns(frequent_itemsets,data_length)
    rules = []
    fpg.rules_generator(frequentPatterns, min_confidence, rules)
    filter_rules = print_rules(rules)

    point_result = process_scatter_one(result,rules,data_length)

    all_node, all_link = get_node2(filter_rules)

    return dataset,all_node, all_link,point_result

# 处理一个散点图
def process_scatter_one(frequent_itemsets,rules,data_length):
    frequent_set_list = []
    confidence_list = []
    result = []

    for itemset, support in frequent_itemsets:
        frequent_set_list.append((itemset, support))

    for (subSet, resSet, confidence) in rules:
        confidence_set = set()
        for item in subSet:
            confidence_set.add(item)
        for item in resSet:
            confidence_set.add(item)

        confidence_list.append((confidence_set,confidence))


    for itemList, support in frequent_set_list:
        itemset = set()
        for item in itemList:
            itemset.add(item)
        for resSet,confidence in confidence_list:

            if itemset.union(resSet) == itemset.intersection(resSet):
                result.append((support/data_length,confidence))

    return result

def process_one_accident_type(result_list,data_set):
    nodes = []
    links = []
    # for accident_type_key in accident_type_keys:
    dataset, all_node, all_link,point_result = process_one_type(result_list,data_set)
    nodes_a, links_a = build_node_and_link( all_node, all_link)
    for node in nodes_a:
        nodes.append(node)
    for link in links_a:
        links.append(link)

    return nodes,links,point_result




def sove_data(accident_type_keys,accident_type_set):
    nodes_return = []
    links_return = []

    x_data = []
    y_data = []
    for index,key in enumerate(accident_type_keys):
        nodes, links,point_result = process_one_accident_type(key, accident_type_set,index)
        for node in nodes:
            nodes_return.append(node )
        for link in links:
            links_return.append(link )

        # 散点图
        for x,y in point_result:
            x_data.append(x)
            y_data.append(y)

    return nodes_return,links_return,x_data,y_data

# 构建一个节点
def build_node(name, value, symbolSize, color):
    node_dic = {
        "name": name,
        "value": value,
        "symbolSize": symbolSize,
        "itemStyle": {
            "color": color
        }
    }
    return node_dic

def save_data(rules,file_name):
    with open(file_name, 'w', encoding="utf-8") as f:
        for (from_items, to_items, confidients) in rules:
            # print("{} -> {} -> {}".format(set(from_items), set(to_items), confidients))
            line = "{} -> {} -> {}".format(set(from_items), set(to_items), confidients)
            f.write(line)
            f.write(os.linesep)
    print("保存结果")


if __name__ == '__main__':

    save_file_name = 'data_rule_002.txt'
    data_file = '/home/sl/workspace/python/a2020/ml-learn/data/test/analysis_1025.csv'
    accident_type_index = 3
    accident_consequence_index = 4
    min_support = 0.05
    min_confidence = 0.8
    file_name = "graph_accident_type_33.html"
    scatter_file_name = "graph_scatter_accident_type_0.html"

    result_list, accident_type_set,data_set = read_data.read_data_all_v2(data_file)

    row_node_set = set()
    row_node_dic = {}
    for row in result_list:
        # print(row)
        for item in row:
            row_node_set.add(item)
            if item not in row_node_dic:
                row_node_dic[item] = 1
            else:
                row_node_dic[item] += 1


    a_node = {}
    a_link = []

    nodes_set = set()
    nodes = []
    links = []

    for index,key in enumerate(data_set.keys()):
        print()
        value_set = data_set[key]
        a_node[key] = len(value_set)
        name = key
        if key  in nodes_set:
            name = key+"1"
        nodes_set.add(name)

        val = 0
        for v in value_set:
            val += row_node_dic[v]

        node_dic = build_node(name,val,val/50, colors[index % len(colors)])
        nodes.append(node_dic)
        # print(node_dic)

        for v in value_set:
            name_v = v
            if v in nodes_set:
                name_v = v + "1"
            nodes_set.add(name_v)
            v_value = row_node_dic[v]
            node_dic2 = build_node(name_v, v_value, v_value/10, colors[index % len(colors)])

            if len(v) > 0:
                # print(node_dic2)
                nodes.append(node_dic2)
                a_node[name] = 5
                a_link.append((name,name_v,1))
                links.append(opts.GraphLink(source=name, target=name_v, value=1, symbol_size=1))

    res_set = row_node_set.difference(nodes_set)
    for index,name in enumerate(res_set):
        # print("{} -{} ".format(name,row_node_dic[name]))
        node_dic = build_node(name, row_node_dic[name],row_node_dic[name]/10, colors[index % len(colors)])
        nodes.append(node_dic)


    # 处理频繁项集
    data_length = len(result_list)
    minimum_support = min_support * data_length
    print("支持度大小: {} -> 总的数据量：{} ->设置的支持度大小：{}".format(minimum_support, data_length, min_support))
    frequent_itemsets = fpg.find_frequent_itemsets(result_list, minimum_support=minimum_support, include_support=True)
    result, frequentPatterns,link_list = print_frequent_patterns(frequent_itemsets, data_length)

    for link in link_list:
        links.append(link)
    # nodes,links = build_node_and_link_v2( a_node, a_link)

    print("数据类型大小：{}".format(len(result_list)))

    rules = []
    fpg.rules_generator(frequentPatterns, min_confidence, rules)
    # filter_rules = print_rules(rules)
    filter_rules = print_rules_v2(rules)
    save_data(rules, save_file_name)

    point_result = process_scatter_one(result, rules, data_length)

    print(row_node_dic)

    # nodes,links,point_result = process_one_accident_type(result_list,data_set)

    c = (
        Graph(init_opts=opts.InitOpts(width="1080px", height="800px"))
            .add("", nodes, links, repulsion=400,
                 is_roam=True,
                 is_focusnode=True,
                 # layout="circular",
                 label_opts=opts.LabelOpts(is_show=True)
                 # linestyle_opts=opts.LineStyleOpts(width=0.5, curve=0.3, opacity=0.7)
                 )
            .set_global_opts(title_opts=opts.TitleOpts(title="Graph-关联规则"))
            .render(file_name)
            # .render("graph_type_0-{}.html".format(accident_type_keys[0]))
    )

