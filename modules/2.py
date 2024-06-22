# _*_ coding : utf-8 _*_
# @Time : 2024/6/14 13:35
# @Author : wfr
# @file : 2
# @Project : IDEA

"""
社团划分测试
"""

from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# # load the karate club graph
# G = nx.karate_club_graph()
# # compute the best partition
# partition = community_louvain.best_partition(G)
data_name = "UAV_GM_360_r=300"
edge_seq_list = np.load('../data/UAV_data/%s_edge_seq.npy' % data_name, allow_pickle=True)
feat = np.load('../data/UAV_data/%s_feat.npy' % data_name, allow_pickle=True)
# 创建一个新的无向图
G = nx.Graph()
# 添加节点特征
for i, f in enumerate(feat):
    G.add_node(i, feature=f)
# 添加边和权重
for edge in edge_seq_list[90]:
    node1, node2, weight = edge
    G.add_edge(node1, node2, weight=weight)

# 返回一个字典，key为节点编号，value为节点所属社团编号
partition = community_louvain.best_partition(G)

# draw the graph
pos = nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()
