# _*_ coding : utf-8 _*_
# @Time : 2024/7/4 13:28
# @Author : wfr
# @file : preprocess
# @Project : IDEA

import torch
import pickle
from my_modules.utils import *
from my_modules.loss import *
import scipy.sparse
import random
import numpy as np
import networkx as nx
import torch_geometric as tg
from torch_geometric.data import Data
import scipy.sparse as sp
from community import community_louvain


def setup_seed(seed):
    torch.manual_seed(seed)  # 设置PyTorch随机种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch的CUDA随机种子（如果GPU可用）
    np.random.seed(seed)  # 设置NumPy的随机种子
    random.seed(seed)  # 设置Python内置random库的随机种子
    torch.backends.cudnn.deterministic = True  # 设置使用CUDA加速时保证结果一致性


setup_seed(0)

# ================
data_name = 'GM_2000_4'
num_nodes = 100  # Number of nodes
num_snaps = 180  # Number of snapshots
# ================

edge_seq_list = np.load('../data/UAV_data/%s_edge_seq.npy' % data_name, allow_pickle=True)
edge_seq_list = edge_seq_list[0:180]

# =========
max_thres = 0  # Threshold for maximum edge weight
for i in range(len(edge_seq_list)):
    for j in range(len(edge_seq_list[i])):
        max_thres = max(edge_seq_list[i][j][2], max_thres)

# ==================
feat = np.load('../data/UAV_data/%s_feat.npy' % data_name, allow_pickle=True)
feat_list = []
for i in range(num_snaps):
    adj = get_adj_wei(edge_seq_list[i], num_nodes, max_thres)
    feat_list.append(np.concatenate((feat, adj), axis=1))

# ==========
pyg_graphs = []
for i in range(num_snaps):
    # ============
    edge_seq = edge_seq_list[i]
    adj = get_adj_norm_wei_with_self_loop(edge_seq, num_nodes, max_thres)
    adj_sp = sp.coo_matrix(adj, dtype=np.float32)
    rowsum = np.array(adj_sp.sum(1), dtype=np.float32)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
    adj_normalized = adj_sp.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj_normalized)
    # =============
    feat = feat_list[i]
    rowsum = np.array(feat.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    feat = r_mat_inv.dot(feat)
    x = torch.FloatTensor(feat)
    # ==============
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    pyg_graphs.append(data)

# 保存
with open('../pyg_graphs/%s_pyg_graphs.pkl' % data_name, 'wb') as f:
    pickle.dump(pyg_graphs, f)
