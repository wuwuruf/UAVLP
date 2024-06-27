# _*_ coding : utf-8 _*_
# @Time : 2024/4/19 20:15
# @Author : wfr
# @file : gen_UAV_topo
# @Project : IDEA

import pandas as pd
import numpy as np
import pickle
from utils import *
import matplotlib.pyplot as plt
import math

# 参数设置
r = 2000  # 通信半径
data_name = 'GM_2000_WEI'
# ========================================
# ========== 构建edge_seq===============
# 读取csv文件
df = pd.read_csv('GM_location_360s.csv')
df_SNR = pd.read_csv('snr_data.csv')

edge_seq = []


# 计算边的权重函数
# def calculate_weight(distance, SNR):
#     if distance < r:
#         return r * math.log2(1 + SNR) / distance
#     else:
#         return 0
# ==========================
# 计算边的权重函数
def calculate_weight(distance, SNR):
    if distance < r:
        return (r - distance) * math.log2(1 + SNR) / r
    else:
        return 0


# 遍历每个时间点
for time_point in df['time'].unique():
    snapshot = df[df['time'] == time_point]
    num_drones = len(snapshot)

    edge_list = []

    # 计算每对无人机之间的距离并生成边
    for i in range(num_drones):
        for j in range(i + 1, num_drones):
            drone1 = snapshot.iloc[i]
            drone2 = snapshot.iloc[j]

            distance = np.sqrt((drone1['x'] - drone2['x']) ** 2 +
                               (drone1['y'] - drone2['y']) ** 2)

            SNR = df_SNR.loc[(df_SNR['i'] == i) & (df_SNR['j'] == j), 'SNR']
            weight = calculate_weight(distance, SNR)
            if weight != 0:
                edge_list.append((int(drone1['ID']), int(drone2['ID']), weight))

    edge_seq.append(edge_list)

np.save('%s_edge_seq.npy' % data_name, np.array(edge_seq, dtype=object))
# ====================================================


# =======构建feat===========
# # 创建一个空的feat数组，数据类型为float64，形状为（100，32）
# feat = np.zeros((100, 32), dtype=np.float64)
#
# # IP地址范围从10.0.1.0到10.0.100.0
# start_ip = 167772160  # 对应10.0.1.0的整数表示
# end_ip = 168034304  # 对应10.0.100.0的整数表示
#
# # 将IP地址转换为32位二进制形式，并存储到feat数组中
# for i in range(100):
#     ip_int = start_ip + (i + 1) * 256  # 每个节点之间的IP地址差为256
#     ip_bin = format(ip_int, '032b')
#     feat[i] = np.array([int(bit) for bit in ip_bin])
#
# # 保存feat数组为npy文件
# np.save('%s_feat.npy' % data_name, feat)
# ====================================================


# # ================================================
# # ======构建模块化矩阵==========这段代码没问题
# # 从npy文件中读取edge_seq
# edge_seq = np.load('UAV_GM_360_r=300_edge_seq.npy', allow_pickle=True)
#
# n = 100
# max_thres = 1
# num_snaps = 361
#
# # 初始化mod数组
# mod = np.zeros((num_snaps, n, n), dtype=np.float32)
#
# # 计算每个图快照的模块化矩阵Q并存储到mod数组中
# for i, snapshot_edges in enumerate(edge_seq):
#     adj = get_adj_wei(snapshot_edges, n, max_thres)
#     adj = adj / max_thres  # 先标准化再计算mod
#
#     degrees = np.sum(adj, axis=1)
#     W = np.sum(adj) / 2
#
#     Q = np.zeros((n, n), dtype=np.float32)
#     for u in range(n):
#         for v in range(n):
#             k_u = degrees[u]
#             k_v = degrees[v]
#             Q[u, v] = adj[u, v] - k_u * k_v / (2 * W)
#
#     mod[i] = Q
#
# # 将mod保存到npy文件中
# np.save('UAV_GM_360_r=300_mod.npy', mod)
# # # ====================================================


# ========================================
# ========查看边数量变化情况===============
# 从npy文件中加载edge_seq
# edge_seq = np.load('%s_edge_seq.npy' % data_name, allow_pickle=True)

# # 计算每张图快照的边数量
# num_edges = [len(edges) for edges in edge_seq]
#
# # 绘制图像
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, len(num_edges) + 1), num_edges, color='b', linestyle='-')
#
# # 添加标题和标签
# plt.title('Number of Edges in Each Snapshot')
# plt.xlabel('Snapshot Number')
# plt.ylabel('Number of Edges')
#
# # 显示网格线
# plt.grid(True)
# plt.savefig(f'edge_num_GM_2000.png')
#
# # 显示图形
# plt.show()
