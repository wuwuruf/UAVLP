# _*_ coding : utf-8 _*_
# @Time : 2024/4/5 22:28
# @Author : wfr
# @file : 1
# @Project : IDEA

# import datetime
# current_time = datetime.datetime.now().time().strftime("%H:%M:%S")
# print(current_time)
#
# import torch
#
# print(torch.cuda.is_available())
#
# position_inputs = torch.arange(0, 10)
# position_inputs = position_inputs.reshape(1, -1)
# position_inputs = position_inputs.repeat(8, 1).long()
# print(1)


import numpy as np


# 读取npy文件
# data = np.load('../data/UAV_data/UAV_RPGM_360_r=500_edge_seq.npy', allow_pickle=True)
data = np.load('../emb_DySAT/emb_DySAT_UAV_RPGM_360_r=300.npy', allow_pickle=True)

# 显示npy文件中的内容
print("Numpy数组的形状：", data.shape)
print("Numpy数组的数据类型：", data.dtype)

print(data[1])
print(data[2])
print(data[3])

# 计算最大边权重
max_weight = 0
for i in range(len(data)):
    for j in range(len(data[i])):
        max_weight = max(data[i][j][1], max_weight)
print("最大边权重为：", max_weight)

# 计算平均边密度
total_density = 0
for i in range(data.shape[0]):
    edge_list = data[i]

    # 计算实际存在的边数
    actual_edges = len(edge_list)

    # 计算节点数，假设节点编号从1开始
    nodes = set()
    for edge in edge_list:
        nodes.add(edge[0])
        nodes.add(edge[1])
    num_nodes = len(nodes)
    # print("图快照%d的节点数为%d" % (i, num_nodes))

    # 计算完全图的边数
    complete_edges = num_nodes * (num_nodes - 1) / 2

    # 计算图的边密度
    density = actual_edges / complete_edges
    total_density += density

av_density = total_density / data.shape[0]
print("图的边密度为:", av_density)

emb_list = np.load('../emb_Node2Vec/emb_Node2Vec_UAV_GM_r=400.npy')
F2 = np.linalg.norm(emb_list[0][0])
print(emb_list[0][0], F2)