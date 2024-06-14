# _*_ coding : utf-8 _*_
# @Time : 2024/6/13 22:52
# @Author : wfr
# @file : MultiAggLP_demo
# @Project : IDEA

import torch
import torch.optim as optim
import torch_geometric as tg
from my_modules.model import MultiAggLP
from my_modules.utils import *
from my_modules.loss import *
import scipy.sparse
import random
import numpy as np
import networkx as nx
from community import community_louvain
import datetime

device = torch.device('cuda')


def setup_seed(seed):
    torch.manual_seed(seed)  # 设置PyTorch随机种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch的CUDA随机种子（如果GPU可用）
    np.random.seed(seed)  # 设置NumPy的随机种子
    random.seed(seed)  # 设置Python内置random库的随机种子
    torch.backends.cudnn.deterministic = True  # 设置使用CUDA加速时保证结果一致性


setup_seed(0)

# =================
data_name = 'UAV_RPGM_360_r=300'
num_nodes = 100  # Number of nodes
num_snaps = 361  # Number of snapshots
max_thres = 100  # Threshold for maximum edge weight
feat_dim = 32  # Dimensionality of node feature
GAT_output_dim = 64
micro_dims = [feat_dim, 128, GAT_output_dim]  # 两层GAT的输入维度、隐藏维度，输出维度
pooling_ratio = 0.5
agg_feat_dim = GAT_output_dim
RNN_dims = [agg_feat_dim, 128, agg_feat_dim]  # 两层GRU的维度
decoder_dims = [RNN_dims[-1], 128, num_nodes]  # 解码器两层全连接的维度
save_flag = False

# =================
dropout_rate = 0.0  # Dropout rate
win_size = 10  # Window size of historical snapshots
epsilon = 0.01  # Threshold of the zero-refining
num_epochs = 100  # Number of training epochs
num_test_snaps = 90  # Number of test snapshots  约7:3划分训练集与测试集 效果不好再改为8:2
num_val_snaps = 20  # Number of validation snapshots
num_train_snaps = num_snaps - num_test_snaps - num_val_snaps  # Number of training snapshots
n_heads = 8

# =================
# loss的超参数
alpha = 20  # Parameter to balance the ER loss
beta = 0.1  # Parameter to balance the SDM loss
lambd = 0.1  # Parameter of attentive aligning unit
theta = 0.1  # Decaying factor

# =================
edge_seq_list = np.load('data/UAV_data/%s_edge_seq.npy' % data_name, allow_pickle=True)
# # 转成合适的格式
edge_index_list = []
edge_weight_list = []
for i in range(num_snaps):
    # 去掉edge_seq中的边权重，并转为适合输入Node2Vec模型的格式
    edge_index = [[], []]
    edge_weight = []
    for edge in edge_seq_list[i]:
        edge_index[0].append(edge[0])
        edge_index[1].append(edge[1])
        edge_weight.append(edge[2])
    edge_index_tnr = torch.LongTensor(edge_index).to(device)
    edge_weight_tnr = torch.FloatTensor(edge_weight).to(device)
    edge_index_list.append(edge_index_tnr)
    edge_weight_list.append(edge_weight_tnr)
feat = np.load('data/UAV_data/%s_feat.npy' % data_name, allow_pickle=True)
feat_tnr = torch.FloatTensor(feat).to(device)  # 放到GPU上
feat_list = []
for i in range(num_snaps):
    feat_list.append(feat_tnr)
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
    # # ================  好像不对，不过上面的没问题了
    # adj_sparse_matrix = nx.to_scipy_sparse_matrix(G, format='coo', nodelist=sorted(G.nodes()))
    # edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj_sparse_matrix)
    # edge_index.to(device)
    # edge_weight.to(device)
    # edge_index_list.append(edge_index)
    # edge_weight_list.append(edge_weight)

# ===================
# 社团划分
partition_dict_list = []
for G in graphs:
    partition_dict = community_louvain.best_partition(G)  # key为节点编号，value为节点所属社团编号
    partition_dict_list.append(partition_dict)
# ==================
# 定义模型和优化器
model = MultiAggLP(micro_dims, pooling_ratio, agg_feat_dim, RNN_dims, decoder_dims, n_heads, dropout_rate).to(device)
opt = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

# ==================
for epoch in range(num_epochs):
    # ============
    # 训练模型
    model.train()
    current_time = datetime.datetime.now().time().strftime("%H:%M:%S")
    # ============
    train_cnt = 0
    loss_list = []
    for tau in range(win_size, num_train_snaps):  # 遍历训练集的所有待预测时刻
        # ================
        # 当前窗口的预测资料
        cur_edge_index_list = edge_index_list[tau - win_size: tau]
        cur_edge_weight_list = edge_weight_list[tau - win_size: tau]  # 这里权重并没有标准化
        cur_feat_list = feat_list[tau - win_size: tau]
        cur_partition_dict_list = partition_dict_list[tau - win_size: tau]
        # ================
        # 获取按社团划分的edge_index和edge_weight
        edge_index_com_list_list = []  # 里面包含了窗口内每张图的每个社团的edge_index（每张图拥有多个edge_index，即每个社团的，组成一个列表）
        edge_weight_com_list_list = []
        for t in range(win_size):  # 遍历窗口内每张图
            partition_dict = cur_partition_dict_list[t]
            edge_index = cur_edge_index_list[t]
            edge_weight = cur_edge_weight_list[t]
            num_coms = max(partition_dict.values()) + 1  # 当前图的社团数量
            edge_index_com_list = [[[], []]] * num_coms  # 当前图的每个社团的edge_index的列表，列表长度等于社团数
            edge_weight_com_list = [[]] * num_coms
            for i in range(len(edge_index[0])):  # 遍历所有边，看其端点是否属于同一社团，若属于则加入对应的edge_index_com
                node1 = edge_index[0][i]
                node2 = edge_index[1][i]
                weight = edge_weight[i]
                if partition_dict[node1.item()] == partition_dict[node2.item()]:
                    com_id = partition_dict[node1.item()]
                    edge_index_com_list[com_id][0].append(node1)
                    edge_index_com_list[com_id][1].append(node2)
                    edge_weight_com_list[com_id].append(weight)
            edge_index_com_list_list.append(edge_index_com_list)
            edge_weight_com_list_list.append(edge_weight_com_list)
        # =================
        # 获取真实邻接矩阵
        gnd_list = []
        for t in range(tau - win_size + 1, tau + 1):
            edges = edge_seq_list
            gnd = get_adj_wei(edges, num_nodes, max_thres)
            gnd_norm = gnd / max_thres  # 这里对邻接矩阵权重进行了标准化
            gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
            gnd_list.append(gnd_tnr)
        # ================
        # 预测及计算损失，反向传播优化参数
        pred_adj_list = model(cur_edge_index_list, cur_edge_weight_list, cur_feat_list, edge_index_com_list_list,
                              edge_weight_com_list_list, pred_flag=False)
        loss = get_loss(pred_adj_list, gnd_list, max_thres, alpha, beta, theta)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # ===============
        loss_list.append(loss.item())
        train_cnt += 1
        if train_cnt % 100 == 0:
            print('-Train %d / %d' % (train_cnt, num_train_snaps))
    loss_mean = np.mean(loss_list)
    print('Epoch#%d Train G-Loss %f' % (epoch, loss_mean))

    # =====================
    # 验证模型
    model.eval()
    # =============
    RMSE_list = []
    MAE_list = []
    MLSD_list = []
    MR_list = []
    for tau in range(num_snaps - num_test_snaps - num_val_snaps, num_snaps - num_test_snaps):  # 遍历验证集的所有待预测时刻
        # ================
        # 当前窗口的预测资料
        cur_edge_index_list = edge_index_list[tau - win_size: tau]
        cur_edge_weight_list = edge_weight_list[tau - win_size: tau]  # 这里权重并没有标准化
        cur_feat_list = feat_list[tau - win_size: tau]
        cur_partition_dict_list = partition_dict_list[tau - win_size: tau]
        # ================
        # 获取按社团划分的edge_index和edge_weight
        edge_index_com_list_list = []  # 里面包含了窗口内每张图的每个社团的edge_index（每张图拥有多个edge_index，即每个社团的，组成一个列表）
        edge_weight_com_list_list = []
        for t in range(win_size):  # 遍历窗口内每张图
            partition_dict = cur_partition_dict_list[t]
            edge_index = cur_edge_index_list[t]
            edge_weight = cur_edge_weight_list[t]
            num_coms = max(partition_dict.values()) + 1  # 当前图的社团数量
            edge_index_com_list = [[], []] * num_coms  # 当前图的每个社团的edge_index的列表，列表长度等于社团数
            edge_weight_com_list = [] * num_coms
            for i in range(len(edge_index[0])):  # 遍历所有边，看其端点是否属于同一社团，若属于则加入对应的edge_index_com
                node1 = edge_index[0][i]
                node2 = edge_index[1][i]
                weight = edge_weight[i]
                if partition_dict[node1.item()] == partition_dict[node2.item()]:
                    com_id = partition_dict[node1.item()]
                    edge_index_com_list[com_id][0].append(node1)
                    edge_index_com_list[com_id][1].append(node2)
                    edge_weight_com_list[com_id].append(weight)
            edge_index_com_list_list.append(edge_index_com_list)
            edge_weight_com_list_list.append(edge_weight_com_list)
        # ================
        # 预测
        pred_adj_list = model(cur_edge_index_list, cur_edge_weight_list, cur_feat_list, edge_index_com_list_list,
                              edge_weight_com_list_list, cur_partition_dict_list, pred_flag=True)
        pred_adj = pred_adj_list[-1]
        # 将预测邻接矩阵值调整为正常值
        if torch.cuda.is_available():  # 张量转为numpy类型
            pred_adj = pred_adj.cpu().data.numpy()
        else:
            pred_adj = pred_adj.data.numpy()
        pred_adj *= max_thres  # Rescale the edge weights to the original value range
        for r in range(num_nodes):
            pred_adj[r, r] = 0
        for r in range(num_nodes):
            for c in range(num_nodes):
                if pred_adj[r, c] <= epsilon:
                    pred_adj[r, c] = 0
        # ===============
        # 获取真实邻接矩阵
        edges = edge_seq_list[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        # ===============
        # 计算评价指标
        RMSE = get_RMSE(pred_adj, gnd, num_nodes)
        MAE = get_MAE(pred_adj, gnd, num_nodes)
        MLSD = get_MLSD(pred_adj, gnd, num_nodes)
        MR = get_MR(pred_adj, gnd, num_nodes)
        # =============
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
        MLSD_list.append(MLSD)
        MR_list.append(MR)

    # =================
    RMSE_mean = np.mean(RMSE_list)
    RMSE_std = np.std(RMSE_list, ddof=1)
    MAE_mean = np.mean(MAE_list)
    MAE_std = np.std(MAE_list, ddof=1)
    MLSD_mean = np.mean(MLSD_list)
    MLSD_std = np.std(MLSD_list, ddof=1)
    MR_mean = np.mean(MR_list)
    MR_std = np.std(MR_list, ddof=1)
    print('Val #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f'
          % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
    # ==========
    f_input = open('res/%s_IDEA_rec.txt' % data_name, 'a+')
    f_input.write('Val #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f Time %s\n'
                  % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std, current_time))
    f_input.close()

# =====================
# 测试模型
model.eval()
current_time = datetime.datetime.now().time().strftime("%H:%M:%S")
# =============
RMSE_list = []
MAE_list = []
MLSD_list = []
MR_list = []
for tau in range(num_snaps - num_test_snaps, num_snaps):  # 遍历测试集的每个tau
    # ================
    # 当前窗口的预测资料
    cur_edge_index_list = edge_index_list[tau - win_size: tau]
    cur_edge_weight_list = edge_weight_list[tau - win_size: tau]  # 这里权重并没有标准化
    cur_feat_list = feat_list[tau - win_size: tau]
    cur_partition_dict_list = partition_dict_list[tau - win_size: tau]
    # ================
    # 获取按社团划分的edge_index和edge_weight
    edge_index_com_list_list = []  # 里面包含了窗口内每张图的每个社团的edge_index（每张图拥有多个edge_index，即每个社团的，组成一个列表）
    edge_weight_com_list_list = []
    for t in range(win_size):  # 遍历窗口内每张图
        partition_dict = cur_partition_dict_list[t]
        edge_index = cur_edge_index_list[t]
        edge_weight = cur_edge_weight_list[t]
        num_coms = max(partition_dict.values()) + 1  # 当前图的社团数量
        edge_index_com_list = [[], []] * num_coms  # 当前图的每个社团的edge_index的列表，列表长度等于社团数
        edge_weight_com_list = [] * num_coms
        for i in range(len(edge_index[0])):  # 遍历所有边，看其端点是否属于同一社团，若属于则加入对应的edge_index_com
            node1 = edge_index[0][i]
            node2 = edge_index[1][i]
            weight = edge_weight[i]
            if partition_dict[node1.item()] == partition_dict[node2.item()]:
                com_id = partition_dict[node1.item()]
                edge_index_com_list[com_id][0].append(node1)
                edge_index_com_list[com_id][1].append(node2)
                edge_weight_com_list[com_id].append(weight)
        edge_index_com_list_list.append(edge_index_com_list)
        edge_weight_com_list_list.append(edge_weight_com_list)
    # ================
    # 预测
    pred_adj_list = model(cur_edge_index_list, cur_edge_weight_list, cur_feat_list, edge_index_com_list_list,
                          edge_weight_com_list_list, pred_flag=True)
    pred_adj = pred_adj_list[-1]
    # 将预测邻接矩阵值调整为正常值
    if torch.cuda.is_available():  # 张量转为numpy类型
        pred_adj = pred_adj.cpu().data.numpy()
    else:
        pred_adj = pred_adj.data.numpy()
    pred_adj *= max_thres  # Rescale the edge weights to the original value range
    for r in range(num_nodes):
        pred_adj[r, r] = 0
    for r in range(num_nodes):
        for c in range(num_nodes):
            if pred_adj[r, c] <= epsilon:
                pred_adj[r, c] = 0
    # ===============
    # 获取真实邻接矩阵
    edges = edge_seq_list[tau]
    gnd = get_adj_wei(edges, num_nodes, max_thres)
    # ===============
    # 计算评价指标
    RMSE = get_RMSE(pred_adj, gnd, num_nodes)
    MAE = get_MAE(pred_adj, gnd, num_nodes)
    MLSD = get_MLSD(pred_adj, gnd, num_nodes)
    MR = get_MR(pred_adj, gnd, num_nodes)
    # =============
    RMSE_list.append(RMSE)
    MAE_list.append(MAE)
    MLSD_list.append(MLSD)
    MR_list.append(MR)

# =================
RMSE_mean = np.mean(RMSE_list)
RMSE_std = np.std(RMSE_list, ddof=1)
MAE_mean = np.mean(MAE_list)
MAE_std = np.std(MAE_list, ddof=1)
MLSD_mean = np.mean(MLSD_list)
MLSD_std = np.std(MLSD_list, ddof=1)
MR_mean = np.mean(MR_list)
MR_std = np.std(MR_list, ddof=1)
print('Test RMSE %f %f MAE %f %f MLSD %f %f MR %f %f'
      % (RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
# ==========
f_input = open('res/%s_IDEA_rec.txt' % data_name, 'a+')
f_input.write('Test RMSE %f %f MAE %f %f MLSD %f %f MR %f %f Time %s\n'
              % (RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std, current_time))
f_input.close()
