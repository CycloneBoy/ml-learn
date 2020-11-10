#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : analysis_solve.py
# @Author: sl
# @Date  : 2020/10/25 - 下午9:23

import datamining.association_rules.fp_growth_first as fpg
import datamining.association_rules.read_data as read_data


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


def get_node(frequent_itemsets,index):
    node_dic = {}
    node_set = set()
    for itemset, support in frequent_itemsets:  # 将generator结果存入list
        for item in itemset:
            node_set.add(item )
            node_dic[item] = "{}_{}".format(item,index)

    all_node = {}
    all_link = []

    for node in node_set:
        all_node[node_dic[node]] = 0

    # todo
    for itemset, support in frequent_itemsets:
        for item in itemset:
            all_node[node_dic[item]] = all_node[node_dic[item]] + support
        #　边
        if len(itemset) >=2:
            first = itemset[0]
            for idx ,item in enumerate(itemset[1:]):
                all_link.append((node_dic[first],node_dic[item],support))
                first = item

    # for key in all_node.keys():
    #     all_node[node_dic[key]] = al

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
def print_frequent_patterns(frequent_itemsets,data_length):
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
def build_node_and_link(accident_type_key,index,all_node,all_link):
    nodes = []
    links = []

    all_node,node_size = node_normalize(all_node)

    node_dic = {
        "name": accident_type_key,
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
        links.append(opts.GraphLink(source=accident_type_key, target=key, value=all_node[key]))
    for first, item, support in all_link:
        links.append(opts.GraphLink(source=first, target=item, value=support / 10, symbol_size=support / 10))

    return nodes,links


# 处理一种类别
def process_one_type(accident_type_key, accident_type_set,index):
    dataset = accident_type_set[accident_type_key]
    data_length = len(dataset)
    minimum_support = min_support * data_length
    print("支持度大小: {} -> 总的数据量：{} ->设置的支持度大小：{}".format(minimum_support, data_length, min_support))
    frequent_itemsets = fpg.find_frequent_itemsets(dataset, minimum_support=minimum_support, include_support=True)
    result, frequentPatterns = print_frequent_patterns(frequent_itemsets,data_length)
    all_node, all_link = get_node(result,index)
    rules = []
    fpg.rules_generator(frequentPatterns, min_confidence, rules)
    print_rules(rules)

    point_result = process_scatter_one(result,rules,data_length)

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

def process_one_accident_type(accident_type_key,accident_type_set,index):
    nodes = []
    links = []
    # for accident_type_key in accident_type_keys:
    dataset, all_node, all_link,point_result = process_one_type(accident_type_key, accident_type_set,index)
    nodes_a, links_a = build_node_and_link(accident_type_key, index, all_node, all_link)
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

if __name__ == '__main__':

    data_file = '/home/sl/workspace/python/a2020/ml-learn/data/test/analysis_1025.csv'
    accident_type_index = 3
    accident_consequence_index = 4
    min_support = 0.2
    min_confidence = 0.8
    file_name =""
    scatter_file_name =""
    # run_type = accident_consequence_index
    run_type = accident_type_index
    if run_type == 3:
        file_name = "graph_accident_type_0.html"
        scatter_file_name = "graph_scatter_accident_type_0.html"
    else:
        file_name = "graph_accident_consequence.html"
        scatter_file_name = "graph_scatter_accident_consequence.html"


    accident_type_keys, accident_type_set = read_data.read_data_by_type(data_file, run_type)
    print("类型大小：{}".format(len(accident_type_keys)))

    # index = 1
    # nodes,links = process_one_accident_type(accident_type_keys[index],accident_type_set,index)
    #
    index = 0
    # nodes,links,point_result = process_one_accident_type(accident_type_keys[index],accident_type_set,index)

    nodes,links,x_data,y_data = sove_data(accident_type_keys,accident_type_set)

    print("graph_{}.html".format(accident_type_keys[0]))
    c = (
        Graph(init_opts=opts.InitOpts(width="1080px", height="800px"))
            .add("", nodes, links, repulsion=400,
                 is_roam=True,
                 is_focusnode=True,
                 # layout="circular",
                 label_opts=opts.LabelOpts(is_show=False)
                 # linestyle_opts=opts.LineStyleOpts(width=0.5, curve=0.3, opacity=0.7)
                 )
            .set_global_opts(title_opts=opts.TitleOpts(title="Graph-"+ accident_type_keys[0]))
            .render(file_name)
            # .render("graph_type_0-{}.html".format(accident_type_keys[0]))
    )

    # x_data = []
    # y_data = []
    # for x,y in point_result:
    #     x_data.append(x)
    #     y_data.append(y)

    # 绘制散点图
    s = (
    Scatter(init_opts=opts.InitOpts(width="800px", height="600px"))
    .add_xaxis(
                xaxis_data=x_data)
    .add_yaxis(
        series_name="confidence",
        y_axis=y_data,
        symbol_size=5,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_series_opts()
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(
            name="support",
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        yaxis_opts=opts.AxisOpts(
            name="confidence",
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        tooltip_opts=opts.TooltipOpts(is_show=True),
        title_opts=opts.TitleOpts(title="scatter-" + accident_type_keys[0])
    )
    .render(scatter_file_name)
)



