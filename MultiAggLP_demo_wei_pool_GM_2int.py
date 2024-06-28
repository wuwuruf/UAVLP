# _*_ coding : utf-8 _*_
# @Time : 2024/6/13 22:52
# @Author : wfr
# @file : MultiAggLP_demo
# @Project : IDEA

import torch
import torch.optim as optim
import torch_geometric as tg
from my_modules.model_wei_pool import MultiAggLP
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

data_name = 'GM_2000_2'
num_nodes = 100  # Number of nodes
num_snaps = 180  # Number of snapshots
max_thres = 100  # Threshold for maximum edge weight
feat_dim = 132  # Dimensionality of node feature
GAT_output_dim = 128
micro_dims = [feat_dim, 128, GAT_output_dim]  # 两层GAT的输入维度、隐藏维度，输出维度
pooling_ratio = 0.8
agg_feat_dim = GAT_output_dim
RNN_dims = [agg_feat_dim, 256, 256]  # 两层GRU的维度
decoder_dims = [RNN_dims[-1], 256, num_nodes]  # 解码器两层全连接的维度
save_flag = False

# =================
dropout_rate = 0.5  # Dropout rate
win_size = 10  # Window size of historical snapshots
epsilon = 0.01  # Threshold of the zero-refining
num_epochs = 1000  # Number of training epochs
num_test_snaps = 20  # Number of test snapshots  约7:3划分训练集与测试集 效果不好再改为8:2
num_val_snaps = 10  # Number of validation snapshots
num_train_snaps = num_snaps - num_test_snaps - num_val_snaps  # Number of training snapshots
n_heads = 8

# =================
# loss的超参数
lambd_cross = 5
lambd_reg = 0.001
theta = 0.2  # Decaying factor
sparse_beta = 10

# =================
edge_seq_list = np.load('data/UAV_data/%s_edge_seq.npy' % data_name, allow_pickle=True)
edge_seq_list = edge_seq_list[0:180]
# # 转成合适的格式
edge_index_list = []
edge_weight_list = []
D_list = []
for i in range(num_snaps):
    # 去掉edge_seq中的边权重，并转为适合输入Node2Vec模型的格式
    edge_index = [[], []]
    edge_weight = []
    for edge in edge_seq_list[i]:  # 每条边代表的是无向边！！不存在重复
        edge_index[0].append(edge[0])
        edge_index[1].append(edge[1])
        edge_weight.append(edge[2])  # 输入GAT前权重归一化
    D = get_D_by_edge_index_and_weight(edge_index, edge_weight, num_nodes)
    D_tnr = torch.FloatTensor(D).to(device)
    edge_index_tnr = torch.LongTensor(edge_index).to(device)
    edge_weight_tnr = torch.FloatTensor(edge_weight).to(device)
    edge_index_list.append(edge_index_tnr)
    edge_weight_list.append(edge_weight_tnr)
    D_list.append(D_tnr)
feat = np.load('data/UAV_data/%s_feat.npy' % data_name, allow_pickle=True)
feat_tnr = torch.FloatTensor(feat).to(device)  # 放到GPU上
feat_list = []
for i in range(num_snaps):
    adj = get_adj_wei(edge_seq_list[i], num_nodes, max_thres)
    adj_tnr = torch.FloatTensor(adj).to(device)
    feat_list.append(torch.cat([feat_tnr, adj_tnr], dim=1))
data_name = 'GM_2000_2_180'
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
model = MultiAggLP(micro_dims, agg_feat_dim, RNN_dims, decoder_dims, n_heads, dropout_rate).to(device)
opt = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)

# ==================
for epoch in range(num_epochs):
    # ============
    # 训练模型
    model.train()
    current_time = datetime.datetime.now().time().strftime("%H:%M:%S")
    # ============
    train_cnt = 0
    loss_list = []
    # =======================
    # 每个epoch的打乱顺序都不同
    random.seed(epoch)
    # 生成从win_size到num_train_snaps范围内的列表
    indices = list(range(win_size, num_train_snaps))
    # 对列表进行随机打乱
    random.shuffle(indices)
    # =======================
    step_interval = 10
    iteration_count = 0
    # =================
    for tau in indices:  # 遍历训练集的所有待预测时刻
        # ================
        # 当前窗口的预测资料
        cur_edge_index_list = edge_index_list[tau - win_size: tau]
        cur_edge_weight_list = edge_weight_list[tau - win_size: tau]  # 这里权重并没有标准化
        cur_feat_list = feat_list[tau - win_size: tau]
        cur_partition_dict_list = partition_dict_list[tau - win_size: tau]
        # ===================
        cur_D_list = D_list[tau - win_size: tau]
        cur_D_com_list_list = D_com_list_list[tau - win_size: tau]
        # ================
        # 获取真实邻接矩阵
        gnd_list = []
        for t in range(tau - win_size + 1, tau + 1):
            edges = edge_seq_list[t]
            gnd = get_adj_no_wei(edges, num_nodes)  # 不加权
            gnd_tnr = torch.FloatTensor(gnd).to(device)
            gnd_list.append(gnd_tnr)
        # ================
        # 预测及计算损失，反向传播优化参数
        pred_adj_list = model(cur_edge_index_list, cur_edge_weight_list, cur_feat_list,
                              cur_D_com_list_list, cur_partition_dict_list, cur_D_list, pred_flag=False)
        iteration_count += 1
        loss = get_corss_reg_loss(sparse_beta, gnd_list, pred_adj_list, theta, lambd_cross, lambd_reg)
        loss.backward()  # 累积梯度

        if iteration_count % step_interval == 0:
            opt.step()
            opt.zero_grad()
            iteration_count = 0

        # ===============
        loss_list.append(loss.item())
        train_cnt += 1
        # if train_cnt % 20 == 0:
        #     print('-Train %d / %d' % (train_cnt, num_train_snaps))
    # 在最后一次循环后，确保执行梯度更新
    if iteration_count % step_interval != 0:
        opt.step()
        opt.zero_grad()
    loss_mean = np.mean(loss_list)
    print('Epoch#%d Train G-Loss %f' % (epoch, loss_mean))

    if save_flag:
        torch.save(model, 'my_pt/MultiAggLP_weipool_lossadd_lossadd_%d.pkl' % epoch)
    # =====================
    # 验证模型
    model.eval()
    # =============
    # RMSE_list = []
    # MAE_list = []
    # MLSD_list = []
    # MR_list = []
    AUC_list = []
    f1_score_list = []
    precision_list = []
    recall_list = []
    for tau in range(num_snaps - num_test_snaps - num_val_snaps, num_snaps - num_test_snaps):  # 遍历验证集的所有待预测时刻
        # ================
        # 当前窗口的预测资料
        cur_edge_index_list = edge_index_list[tau - win_size: tau]
        cur_edge_weight_list = edge_weight_list[tau - win_size: tau]  # 这里权重并没有标准化
        cur_feat_list = feat_list[tau - win_size: tau]
        cur_partition_dict_list = partition_dict_list[tau - win_size: tau]
        # ===================
        cur_D_list = D_list[tau - win_size: tau]
        cur_D_com_list_list = D_com_list_list[tau - win_size: tau]
        # ================
        # 预测
        pred_adj_list = model(cur_edge_index_list, cur_edge_weight_list, cur_feat_list,
                              cur_D_com_list_list, cur_partition_dict_list, cur_D_list, pred_flag=True)
        pred_adj = pred_adj_list[-1]
        # ===========================
        # 以下是不加权二分类预测
        # ===========================
        pred_adj = pred_adj.cpu().data.numpy()  # 转为numpy
        for r in range(num_nodes):
            pred_adj[r, r] = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                pre_av = (pred_adj[i, j] + pred_adj[j, i]) / 2
                pred_adj[i, j] = pre_av
                pred_adj[j, i] = pre_av
        # ==============
        # 获取真实的不加权0-1邻接矩阵
        edges = edge_seq_list[tau]
        gnd = get_adj_no_wei(edges, num_nodes)
        # ===============
        # 计算评价指标
        AUC = get_AUC(pred_adj, gnd)
        f1_score = get_f1_score(pred_adj, gnd)
        precision = get_precision_score(pred_adj, gnd)
        recall = get_recall_score(pred_adj, gnd)
        # ===============
        AUC_list.append(AUC)
        f1_score_list.append(f1_score)
        precision_list.append(precision)
        recall_list.append(recall)
    # ==============
    AUC_mean = np.mean(AUC_list)
    AUC_std = np.std(AUC_list, ddof=1)
    f1_score_mean = np.mean(f1_score_list)
    f1_score_std = np.std(f1_score_list, ddof=1)
    precision_mean = np.mean(precision_list)
    precision_std = np.std(precision_list, ddof=1)
    recall_mean = np.mean(recall_list)
    recall_std = np.std(recall_list, ddof=1)
    # ==============
    print('Val #%d AUC %f %f f1_score %f %f precision %f %f recall %f %f'
          % (epoch, AUC_mean, AUC_std, f1_score_mean, f1_score_std, precision_mean, precision_std, recall_mean,
             recall_std))
    # ==========
    f_input = open('res/%s_MultiAggLP_norm_weipool_lossadd_lstm256_binary_rec.txt' % data_name, 'a+')
    f_input.write('Val #%d Loss %f AUC %f %f f1_score %f %f precision %f %f recall %f %f Time %s\n'
                  % (epoch, loss_mean, AUC_mean, AUC_std, f1_score_mean, f1_score_std,
                     precision_mean, precision_std, recall_mean, recall_std, current_time))
    f_input.close()
    if (epoch + 1) % 5 == 0:
        # =====================
        # 测试模型
        # model = torch.load('my_pt/MultiAggLP_new_agg_binary_70.pkl')
        model.eval()
        current_time = datetime.datetime.now().time().strftime("%H:%M:%S")
        # =============
        # RMSE_list = []
        # MAE_list = []
        # MLSD_list = []
        # MR_list = []
        AUC_list = []
        f1_score_list = []
        precision_list = []
        recall_list = []
        for tau in range(num_snaps - num_test_snaps, num_snaps):  # 遍历测试集的每个tau
            # ================
            # 当前窗口的预测资料
            cur_edge_index_list = edge_index_list[tau - win_size: tau]
            cur_edge_weight_list = edge_weight_list[tau - win_size: tau]  # 这里权重并没有标准化
            cur_feat_list = feat_list[tau - win_size: tau]
            cur_partition_dict_list = partition_dict_list[tau - win_size: tau]
            # ===================
            cur_D_list = D_list[tau - win_size: tau]
            cur_D_com_list_list = D_com_list_list[tau - win_size: tau]
            # ================
            # 预测
            pred_adj_list = model(cur_edge_index_list, cur_edge_weight_list, cur_feat_list,
                                  cur_D_com_list_list, cur_partition_dict_list, cur_D_list, pred_flag=True)
            pred_adj = pred_adj_list[-1]
            # ===========================
            # 以下是不加权二分类预测
            # ===========================
            pred_adj = pred_adj.cpu().data.numpy()  # 转为numpy
            for r in range(num_nodes):
                pred_adj[r, r] = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    pre_av = (pred_adj[i, j] + pred_adj[j, i]) / 2
                    pred_adj[i, j] = pre_av
                    pred_adj[j, i] = pre_av
            # ==============
            # 获取真实的不加权0-1邻接矩阵
            edges = edge_seq_list[tau]
            gnd = get_adj_no_wei(edges, num_nodes)
            # ===============
            # 计算评价指标
            AUC = get_AUC(pred_adj, gnd)
            f1_score = get_f1_score(pred_adj, gnd)
            precision = get_precision_score(pred_adj, gnd)
            recall = get_recall_score(pred_adj, gnd)
            # ===============
            AUC_list.append(AUC)
            f1_score_list.append(f1_score)
            precision_list.append(precision)
            recall_list.append(recall)
        # ==============
        AUC_mean = np.mean(AUC_list)
        AUC_std = np.std(AUC_list, ddof=1)
        f1_score_mean = np.mean(f1_score_list)
        f1_score_std = np.std(f1_score_list, ddof=1)
        precision_mean = np.mean(precision_list)
        precision_std = np.std(precision_list, ddof=1)
        recall_mean = np.mean(recall_list)
        recall_std = np.std(recall_list, ddof=1)
        # ==============
        print('Test AUC %f %f f1_score %f %f precision %f %f recall %f %f'
              % (
                  AUC_mean, AUC_std, f1_score_mean, f1_score_std, precision_mean, precision_std, recall_mean,
                  recall_std))
        # ==========
        f_input = open('res/%s_MultiAggLP_norm_weipool_lossadd_lstm256_binary_rec.txt' % data_name, 'a+')
        f_input.write('Test AUC %f %f f1_score %f %f precision %f %f recall %f %f Time %s\n'
                      % (AUC_mean, AUC_std, f1_score_mean, f1_score_std, precision_mean, precision_std, recall_mean,
                         recall_std, current_time))
        f_input.close()