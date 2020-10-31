#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test1.py
# @Author: sl
# @Date  : 2020/10/28 - 下午8:42

import json

file_name ="/home/sl/workspace/python/a2020/ml-learn/data/test/npmdepgraph.min10.json"

if __name__ == '__main__':
    load_dict = {}
    with open(file_name, 'r') as load_f:
        load_dict = json.load(load_f)
        # print(load_dict)


    # print(load_dict)

    for node in load_dict["nodes"]:
        print(node)
        break


    print()
    nodes = [
        {
            "x": node["x"],
            "y": node["y"],
            "id": node["id"],
            "name": node["label"],
            "symbolSize": node["size"],
            "itemStyle": {"normal": {"color": node["color"]}},
        }
        for node in load_dict["nodes"]
    ]

    # print(nodes)

    node_color = [
        node["color"]
        for node in load_dict["nodes"]
    ]

    color_set = set(node_color)

    for color in color_set:
        print("\"{}\",".format(color))
