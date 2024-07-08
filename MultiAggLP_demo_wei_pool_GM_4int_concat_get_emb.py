# _*_ coding : utf-8 _*_
# @Time : 2024/6/13 22:52
# @Author : wfr
# @file : MultiAggLP_demo
# @Project : IDEA

import torch
import pickle
import torch.optim as optim
import torch_geometric as tg
from my_modules.model_concat_pre_train import MultiAggLP_emb
from my_modules.utils import *
from my_modules.loss import *
import scipy.sparse
import random
import numpy as np
import networkx as nx
from community import community_louvain
import datetime
import os

device = torch.device('cuda')


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def setup_seed(seed):
    torch.manual_seed(seed)  # 设置PyTorch随机种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch的CUDA随机种子（如果GPU可用）
    np.random.seed(seed)  # 设置NumPy的随机种子
    random.seed(seed)  # 设置Python内置random库的随机种子
    torch.backends.cudnn.deterministic = True  # 设置使用CUDA加速时保证结果一致性


setup_seed(0)

data_name = 'GM_2000_4'
num_nodes = 100  # Number of nodes
num_snaps = 180  # Number of snapshots
max_thres = 100  # Threshold for maximum edge weight
feat_dim = 132  # Dimensionality of node feature
GAT_output_dim = 128
micro_dims = [feat_dim, 128, GAT_output_dim]  # 两层GAT的输入维度、隐藏维度，输出维度
pooling_ratio = 0.8
agg_feat_dim = GAT_output_dim * 3
RNN_dims = [agg_feat_dim, 256, 256]  # 两层GRU的维度
decoder_dims = [RNN_dims[-1], 256, num_nodes]  # 解码器两层全连接的维度
save_flag = False

# =================
dropout_rate = 0.5  # Dropout rate
win_size = 10  # Window size of historical snapshots
epsilon = 0.01  # Threshold of the zero-refining
num_epochs = 800  # Number of training epochs
num_test_snaps = 20  # Number of test snapshots  约7:3划分训练集与测试集 效果不好再改为8:2
num_val_snaps = 10  # Number of validation snapshots
num_train_snaps = num_snaps - num_test_snaps - num_val_snaps  # Number of training snapshots
n_heads = 8
# =================
step_interval = 5
early_stop_epochs = 70
# =================
# loss的超参数
lambd_cross = 5
lambd_reg = 0.001
theta = 0.2  # Decaying factor
sparse_beta = 10
emb_lambda1 = 0.1
emb_lambda2 = 0.01

# =================
edge_seq_list = np.load('data/UAV_data/%s_edge_seq.npy' % data_name, allow_pickle=True)
edge_seq_list = edge_seq_list[0:180]
# =================
with open('pyg_graphs/%s_pyg_graphs.pkl' % data_name, 'rb') as f:
    pyg_graphs = pickle.load(f)
D_list = []  # 计算时不考虑自环，这里度是用于加权池化的
edge_index_list = []
edge_weight_list = []
feat_list = []
for i in range(num_snaps):
    pyg_graph = pyg_graphs[i].to(device)
    edge_index = pyg_graph.edge_index
    edge_weight = pyg_graph.edge_weight
    feat = pyg_graph.x
    D = get_D_by_edge_index_and_weight_tnr(pyg_graph.edge_index, pyg_graph.edge_weight, num_nodes).to(device)
    D_list.append(D)
    edge_index_list.append(edge_index)
    edge_weight_list.append(edge_weight)
    feat_list.append(feat)

# ================
data_name = 'GM_2000_4_180'
# ==================
# 创建nx格式的图列表
graphs = []
for edge_seq in edge_seq_list:
    # 创建一个新的无向图
    G = nx.Graph()
    # 添加节点
    for i in range(num_nodes):
        G.add_node(i)
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

# ==================
edge_index_com_list_list = np.load('com_list_list/%s_edge_index_com_list_list.npy' % data_name, allow_pickle=True)
edge_weight_com_list_list = np.load('com_list_list/%s_edge_weight_com_list_list.npy' % data_name,
                                    allow_pickle=True)
D_com_list_list = []
for i in range(len(edge_index_com_list_list)):
    D_com_list = []
    edge_index_com_list = edge_index_com_list_list[i]
    edge_weight_com_list = edge_weight_com_list_list[i]
    for j in range(len(edge_index_com_list)):
        num_nodes_com = 0
        for key, value in partition_dict_list[i].items():
            if value == j:
                num_nodes_com += 1
        D_com = get_D_by_edge_index_and_weight(edge_index_com_list[j], edge_weight_com_list[j], num_nodes_com)
        D_com_tnr = torch.FloatTensor(D_com).to(device)
        D_com_list.append(D_com_tnr)
    D_com_list_list.append(D_com_list)
# ==================
# 定义模型和优化器
model_emb = MultiAggLP_emb(micro_dims, agg_feat_dim, RNN_dims, n_heads, dropout_rate).to(device)
opt_emb = optim.Adam(model_emb.parameters(), lr=5e-4, weight_decay=5e-4)
# model_decoder = MultiAggLP_emb(micro_dims, agg_feat_dim, RNN_dims, decoder_dims, n_heads, dropout_rate).to(device)
# opt_decoder = optim.Adam(model_decoder.parameters(), lr=5e-4, weight_decay=5e-4)
# ==================
best_loss = 1e5
patience_counter = 0
patience = 10
best_emb_list = []
for epoch in range(num_epochs):

    # ============
    # 训练模型
    model_emb.train()
    current_time = datetime.datetime.now().time().strftime("%H:%M:%S")
    # ============
    loss_list = []
    emb_list = []
    # =======================
    iteration_count = 0
    # =================
    for tau in range(0, num_snaps):  # 遍历所有时刻
        # ================
        # 当前的预测资料
        cur_edge_index = edge_index_list[tau]
        cur_edge_weight = edge_weight_list[tau]  # 这里权重并没有标准化
        cur_feat = feat_list[tau - 1]
        cur_partition_dict = partition_dict_list[tau]
        # ===================
        cur_D = D_list[tau]
        cur_D_com_list = D_com_list_list[tau]
        # ================
        # 获取真实邻接矩阵
        edges = edge_seq_list[tau]
        gnd = get_adj_no_wei(edges, num_nodes)  # 不加权
        gnd_tnr = torch.FloatTensor(gnd).to(device)
        # ================
        # 预测及计算损失，反向传播优化参数
        emb = model_emb(cur_edge_index, cur_edge_weight, cur_feat, cur_D_com_list, cur_partition_dict,
                        cur_D)
        iteration_count += 1
        emb_list.append(emb.cpu().data.numpy())
        loss = get_emb_loss(emb, gnd_tnr, emb_lambda1, emb_lambda2)
        loss.backward()  # 累积梯度

        if iteration_count % step_interval == 0:
            opt_emb.step()
            opt_emb.zero_grad()
            iteration_count = 0

        # ===============
        loss_list.append(loss.item())

    # 在最后一次循环后，确保执行梯度更新
    if iteration_count % step_interval != 0:
        opt_emb.step()
        opt_emb.zero_grad()
    loss_mean = np.mean(loss_list)
    print('Epoch#%d Train G-Loss %f' % (epoch, loss_mean))

    if save_flag:
        torch.save(opt_emb, 'my_pt/MultiAggLP_newGAT_%d.pkl' % epoch)
    # =====================
    if loss_mean < best_loss:
        best_loss = loss_mean
        patience_counter = 0
        best_emb_list = emb_list
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        np.save('emb_MultiAggLP/emb_MultiAggLP_%s_dim=128.npy' % data_name, np.array(best_emb_list))
        break
