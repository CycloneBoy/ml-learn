#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : show6.py
# @Author: sl
# @Date  : 2020/10/28 - 下午9:00
import networkx as nx
from networkx.readwrite import json_graph

from pyecharts.charts import Graph
import pyecharts.options as opts

g = nx.Graph()

g.add_node('N1', name='Node 1', symbolSize=50)
g.add_node('N2', name='Node 2', symbolSize=20)
g.add_node('N3', name='Node 3', symbolSize=30)
g.add_edge('N1', 'N2')
g.add_edge('N1', 'N3')

g_data = json_graph.node_link_data(g)

print(g_data)

eg = Graph(init_opts=opts.InitOpts(width="1600px", height="800px"))
eg.add('Devices', nodes=g_data['nodes'], links=g_data['links'],)
print('symbolSize' in eg.options['series'][0])
# eg.show_config()
eg.render("test.html")