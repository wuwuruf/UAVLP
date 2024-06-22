# _*_ coding : utf-8 _*_
# @Time : 2024/6/19 20:34
# @Author : wfr
# @file : com_partition
# @Project : IDEA

import numpy as np
import networkx as nx
from community import community_louvain
import datetime


data_name = 'UAV_RPGM_360_r=300'
num_snaps = 180
max_weight = 100
# =================

edge_seq_list = np.load('../data/UAV_data/%s_edge_seq.npy' % data_name, allow_pickle=True)
edge_seq_list = edge_seq_list[0:180]
# 转成合适的格式
edge_index_list = []
edge_weight_list = []
for i in range(num_snaps):
    # 去掉edge_seq中的边权重，并转为适合输入Node2Vec模型的格式
    edge_index = [[], []]
    edge_weight = []
    for edge in edge_seq_list[i]:  # 每条边代表的是无向边！！不存在重复
        edge_index[0].append(edge[0])
        edge_index[1].append(edge[1])
        edge_weight.append(edge[2] / max_weight)  # 权重归一化
    # edge_index_tnr = torch.LongTensor(edge_index).to(device)
    # edge_weight_tnr = torch.FloatTensor(edge_weight).to(device)
    edge_index_list.append(edge_index)
    edge_weight_list.append(edge_weight)
# ===================
feat = np.load('../data/UAV_data/%s_feat.npy' % data_name, allow_pickle=True)
feat_list = []
for i in range(num_snaps):
    feat_list.append(feat)
data_name = 'UAV_RPGM_180_r=300'
# ==================
# 创建nx格式的图列表
graphs = []
for edge_seq in edge_seq_list:
    # 创建一个新的无向图
    G = nx.Graph()
    # 添加节点特征
    for i, f in enumerate(feat):
        G.add_node(i, feature=f)
    # 添加边和权重
    for edge in edge_seq:
        node1, node2, weight = edge
        G.add_edge(node1, node2, weight=weight)
    # 将图添加到图列表中
    graphs.append(G)

# ===================
# 社团划分
partition_dict_list = []
for G in graphs:
    partition_dict = community_louvain.best_partition(G, random_state=1)  # key为节点编号，value为节点所属社团编号
    partition_dict_list.append(partition_dict)

# ===================
# 获取按社团划分的edge_index和edge_weight
edge_index_com_list_list = []  # 里面包含了窗口内每张图的每个社团的edge_index（每张图拥有多个edge_index，即每个社团的，组成一个列表）
edge_weight_com_list_list = []
for t in range(num_snaps):
    partition_dict = partition_dict_list[t]
    edge_index = edge_index_list[t]
    edge_weight = edge_weight_list[t]
    num_coms = max(partition_dict.values()) + 1  # 当前图的社团数量
    # ======
    edge_index_com_list = []  # 当前图的每个社团的edge_index的列表，列表长度等于社团数
    edge_weight_com_list = []
    for i in range(num_coms):
        edge_index_com_list.append([[], []])
        edge_weight_com_list.append([])
    # ======
    for i in range(len(edge_index[0])):  # 遍历所有边，看其端点是否属于同一社团，若属于则加入对应的edge_index_com
        node1 = edge_index[0][i]
        node2 = edge_index[1][i]
        weight = edge_weight[i]
        if partition_dict[node1] == partition_dict[node2]:
            com_id = partition_dict[node1]
            edge_index_com_list[com_id][0].append(node1)
            edge_index_com_list[com_id][1].append(node2)
            edge_weight_com_list[com_id].append(weight)
    # ==========
    # 为每个社团内的节点重新编号
    edge_index_com_list_new = []
    for com_id in range(num_coms):
        node_ids = [key for key, value in partition_dict.items() if value == com_id]
        node_set = set(node_ids)
        node_map = {node: i for i, node in enumerate(node_set)}  # 映射字典，旧编号到新编号的映射
        edge_index_com_new = [[node_map[node] for node in edge_index_com_list[com_id][0]],
                              [node_map[node] for node in edge_index_com_list[com_id][1]]]
        edge_index_com_list_new.append(edge_index_com_new)
    # ==========
    # edge_index_com_tnr_list = [torch.LongTensor(edge_index_com).to(device) for edge_index_com in
    #                            edge_index_com_list_new]
    # edge_weight_com_tnr_list = [torch.FloatTensor(edge_weight_com).to(device) for edge_weight_com in
    #                             edge_weight_com_list]
    edge_index_com_list_list.append(edge_index_com_list_new)
    edge_weight_com_list_list.append(edge_weight_com_list)

# ========================
# np.save('../com_list_list/%s_edge_index_com_list_list.npy' % data_name, np.array(edge_index_com_list_list))
# np.save('../com_list_list/%s_edge_weight_com_list_list.npy' % data_name, np.array(edge_weight_com_list_list))
np.save('../com_list_list/%s_edge_index_com_list_list.npy' % data_name,
        np.array(edge_index_com_list_list, dtype=object))
np.save('../com_list_list/%s_edge_weight_com_list_list.npy' % data_name,
        np.array(edge_weight_com_list_list, dtype=object))
