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
num_snaps = 180  # Number of snapshots
max_thres = 100  # Threshold for maximum edge weight
feat_dim = 32  # Dimensionality of node feature
GAT_output_dim = 128
micro_dims = [feat_dim, 64, GAT_output_dim]  # 两层GAT的输入维度、隐藏维度，输出维度
pooling_ratio = 0.5
agg_feat_dim = GAT_output_dim
RNN_dims = [agg_feat_dim, agg_feat_dim, agg_feat_dim]  # 两层GRU的维度
decoder_dims = [RNN_dims[-1], 256, num_nodes]  # 解码器两层全连接的维度
save_flag = True


# =================
dropout_rate = 0.5  # Dropout rate！！！！！！！！！！！！！！！
win_size = 10  # Window size of historical snapshots
epsilon = 0.01  # Threshold of the zero-refining
num_epochs = 200  # Number of training epochs
num_test_snaps = 20  # Number of test snapshots  约7:3划分训练集与测试集 效果不好再改为8:2
num_val_snaps = 10  # Number of validation snapshots
num_train_snaps = num_snaps - num_test_snaps - num_val_snaps  # Number of training snapshots
n_heads = 8

# =================
# loss的超参数
alpha = 20  # Parameter to balance the ER loss
beta = 0.1  # Parameter to balance the SDM loss
lambd = 0.1  # Parameter of attentive aligning unit
theta = 0.2  # Decaying factor
sparse_beta = 2

# =================
edge_seq_list = np.load('data/UAV_data/%s_edge_seq.npy' % data_name, allow_pickle=True)
edge_seq_list = edge_seq_list[0:180]
# # 转成合适的格式
edge_index_list = []
edge_weight_list = []
for i in range(num_snaps):
    # 去掉edge_seq中的边权重，并转为适合输入Node2Vec模型的格式
    edge_index = [[], []]
    edge_weight = []
    for edge in edge_seq_list[i]:  # 每条边代表的是无向边！！不存在重复
        edge_index[0].append(edge[0])
        edge_index[1].append(edge[1])
        edge_weight.append(edge[2])
    edge_index_tnr = torch.LongTensor(edge_index).to(device)
    edge_weight_tnr = torch.FloatTensor(edge_weight).to(device)
    edge_index_list.append(edge_index_tnr)
    edge_weight_list.append(edge_weight_tnr)
# =====================
# feat_tnr = torch.randn(100, 32).to(device)
# ========================
feat = np.load('data/UAV_data/%s_feat.npy' % data_name, allow_pickle=True)
feat_tnr = torch.FloatTensor(feat).to(device)  # 放到GPU上
feat_list = []
for i in range(num_snaps):
    feat_list.append(feat_tnr)
# ======================
data_name = 'UAV_RPGM_180_r=300'
# ==================
# emb_dim = 64
# emb_list = np.load('emb_Node2Vec/emb_Node2Vec_%s_dim=%d.npy' % (data_name, emb_dim))
# feat_list = [torch.FloatTensor(emb).to(device) for emb in emb_list]
# ========================
# 创建nx格式的图列表
graphs = []
for t in range(num_snaps-1):
    edge_seq = edge_seq_list[t]
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
# 读取按社团划分的edge_index和edge_weight，并转tensor
edge_index_com_list_list_notnr = np.load('com_list_list/%s_edge_index_com_list_list.npy' % data_name, allow_pickle=True)
edge_weight_com_list_list_notnr = np.load('com_list_list/%s_edge_weight_com_list_list.npy' % data_name,
                                          allow_pickle=True)
edge_index_com_list_list = []
for edge_index_com_list_notnr in edge_index_com_list_list_notnr:
    edge_index_com_list = []
    for edge_index_com_notnr in edge_index_com_list_notnr:
        edge_index_com = torch.LongTensor(edge_index_com_notnr).to(device)
        edge_index_com_list.append(edge_index_com)
    edge_index_com_list_list.append(edge_index_com_list)
edge_weight_com_list_list = []
for edge_weight_com_list_notnr in edge_weight_com_list_list_notnr:
    edge_weight_com_list = []
    for edge_weight_com_notnr in edge_weight_com_list_notnr:
        edge_weight_com = torch.FloatTensor(edge_weight_com_notnr).to(device)
        edge_weight_com_list.append(edge_weight_com)
    edge_weight_com_list_list.append(edge_weight_com_list)
# ==================
# 定义模型和优化器
model = MultiAggLP(micro_dims, pooling_ratio, agg_feat_dim, RNN_dims, decoder_dims, n_heads, dropout_rate).to(device)
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
    for tau in range(win_size, num_train_snaps):  # 遍历训练集的所有待预测时刻
        # ================
        # 当前窗口的预测资料
        cur_edge_index_list = edge_index_list[tau - win_size: tau]
        cur_edge_weight_list = edge_weight_list[tau - win_size: tau]  # 这里权重并没有标准化
        cur_feat_list = feat_list[tau - win_size: tau]
        cur_partition_dict_list = partition_dict_list[tau - win_size: tau]
        # ===================
        cur_edge_index_com_list_list = edge_index_com_list_list[tau - win_size: tau]
        cur_edge_weight_com_list_list = edge_weight_com_list_list[tau - win_size: tau]
        # ====================
        # # 获取真实邻接矩阵
        # gnd_list = []
        # for t in range(tau - win_size + 1, tau + 1):
        #     edges = edge_seq_list[t]
        #     gnd = get_adj_wei(edges, num_nodes, max_thres)
        #     gnd_norm = gnd / max_thres  # 这里对邻接矩阵权重进行了标准化
        #     gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
        #     gnd_list.append(gnd_tnr)
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
        pred_adj_list = model(cur_edge_index_list, cur_edge_weight_list, cur_feat_list, cur_edge_index_com_list_list,
                              cur_edge_weight_com_list_list, cur_partition_dict_list, pred_flag=False)
        # loss = get_loss(pred_adj_list, gnd_list, max_thres, alpha, beta, theta)
        loss = get_refined_loss(sparse_beta, gnd_list, pred_adj_list, theta)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # ===============
        loss_list.append(loss.item())
        train_cnt += 1
        if train_cnt % 20 == 0:
            print('-Train %d / %d' % (train_cnt, num_train_snaps))
    loss_mean = np.mean(loss_list)
    print('Epoch#%d Train G-Loss %f' % (epoch, loss_mean))

    if save_flag:
        torch.save(model, 'my_pt/MultiAggLP_binary_%d.pkl' % epoch)
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
        cur_edge_index_com_list_list = edge_index_com_list_list[tau - win_size: tau]
        cur_edge_weight_com_list_list = edge_weight_com_list_list[tau - win_size: tau]
        # ================
        # 预测
        pred_adj_list = model(cur_edge_index_list, cur_edge_weight_list, cur_feat_list, cur_edge_index_com_list_list,
                              cur_edge_weight_com_list_list, cur_partition_dict_list, pred_flag=True)
        pred_adj = pred_adj_list[-1]
        # ===========================
        # 以下是不加权二分类预测
        # ===========================
        pred_adj = pred_adj.cpu().data.numpy()  # 转为numpy
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
    f_input = open('res/%s_MultiAggLP_binary_rec.txt' % data_name, 'a+')
    f_input.write('Val #%d Loss %f AUC %f %f f1_score %f %f precision %f %f recall %f %f Time %s\n'
                  % (epoch, loss_mean, AUC_mean, AUC_std, f1_score_mean, f1_score_std,
                     precision_mean, precision_std, recall_mean, recall_std, current_time))
    f_input.close()
    #         # 将预测邻接矩阵值调整为正常值
    #         if torch.cuda.is_available():  # 张量转为numpy类型
    #             pred_adj = pred_adj.cpu().data.numpy()
    #         else:
    #             pred_adj = pred_adj.data.numpy()
    #         pred_adj *= max_thres  # Rescale the edge weights to the original value range
    #         for r in range(num_nodes):
    #             pred_adj[r, r] = 0
    #         for r in range(num_nodes):
    #             for c in range(num_nodes):
    #                 if pred_adj[r, c] <= epsilon:
    #                     pred_adj[r, c] = 0
    #         # ===============
    #         # 获取真实邻接矩阵
    #         edges = edge_seq_list[tau]
    #         gnd = get_adj_wei(edges, num_nodes, max_thres)
    #         # ===============
    #         # 计算评价指标
    #         RMSE = get_RMSE(pred_adj, gnd, num_nodes)
    #         MAE = get_MAE(pred_adj, gnd, num_nodes)
    #         MLSD = get_MLSD(pred_adj, gnd, num_nodes)
    #         MR = get_MR(pred_adj, gnd, num_nodes)
    #         # =============
    #         RMSE_list.append(RMSE)
    #         MAE_list.append(MAE)
    #         MLSD_list.append(MLSD)
    #         MR_list.append(MR)
    #
    #     # =================
    #     RMSE_mean = np.mean(RMSE_list)
    #     RMSE_std = np.std(RMSE_list, ddof=1)
    #     MAE_mean = np.mean(MAE_list)
    #     MAE_std = np.std(MAE_list, ddof=1)
    #     MLSD_mean = np.mean(MLSD_list)
    #     MLSD_std = np.std(MLSD_list, ddof=1)
    #     MR_mean = np.mean(MR_list)
    #     MR_std = np.std(MR_list, ddof=1)
    #     print('Val #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f'
    #           % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
    #     # ==========
    #     f_input = open('res/%s_MultiAggLP_rec.txt' % data_name, 'a+')
    #     f_input.write('Val #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f Time %s\n'
    #                   % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std, current_time))
    #     f_input.close()
    if (epoch + 1) % 5 == 0:
        # =====================
        # 测试模型
        # model = torch.load('my_pt/MultiAggLP_binary_18.pkl')
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
            cur_edge_index_com_list_list = edge_index_com_list_list[tau - win_size: tau]
            cur_edge_weight_com_list_list = edge_weight_com_list_list[tau - win_size: tau]
            # ================
            # 预测
            pred_adj_list = model(cur_edge_index_list, cur_edge_weight_list, cur_feat_list,
                                  cur_edge_index_com_list_list,
                                  cur_edge_weight_com_list_list, cur_partition_dict_list, pred_flag=True)
            pred_adj = pred_adj_list[-1]
            # ===========================
            # 以下是不加权二分类预测
            # ===========================
            pred_adj = pred_adj.cpu().data.numpy()  # 转为numpy
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
              AUC_mean, AUC_std, f1_score_mean, f1_score_std, precision_mean, precision_std, recall_mean, recall_std))
        # ==========
        f_input = open('res/%s_MultiAggLP_binary_rec.txt' % data_name, 'a+')
        f_input.write('Test AUC %f %f f1_score %f %f precision %f %f recall %f %f Time %s\n'
                      % (AUC_mean, AUC_std, f1_score_mean, f1_score_std, precision_mean, precision_std, recall_mean,
                         recall_std, current_time))
        f_input.close()
        # =======================================
        # 这些是加权预测
        # =======================================
        #     # 将预测邻接矩阵值调整为正常值
        #     if torch.cuda.is_available():  # 张量转为numpy类型
        #         pred_adj = pred_adj.cpu().data.numpy()
        #     else:
        #         pred_adj = pred_adj.data.numpy()
        #     pred_adj *= max_thres  # Rescale the edge weights to the original value range
        #     for r in range(num_nodes):
        #         pred_adj[r, r] = 0
        #     for r in range(num_nodes):
        #         for c in range(num_nodes):
        #             if pred_adj[r, c] <= epsilon:
        #                 pred_adj[r, c] = 0
        #     # ===============
        #     # 获取真实邻接矩阵
        #     edges = edge_seq_list[tau]
        #     gnd = get_adj_wei(edges, num_nodes, max_thres)
        #     # ===============
        #     # 计算评价指标
        #     RMSE = get_RMSE(pred_adj, gnd, num_nodes)
        #     MAE = get_MAE(pred_adj, gnd, num_nodes)
        #     MLSD = get_MLSD(pred_adj, gnd, num_nodes)
        #     MR = get_MR(pred_adj, gnd, num_nodes)
        #     # =============
        #     RMSE_list.append(RMSE)
        #     MAE_list.append(MAE)
        #     MLSD_list.append(MLSD)
        #     MR_list.append(MR)
        #
        # # =================
        # RMSE_mean = np.mean(RMSE_list)
        # RMSE_std = np.std(RMSE_list, ddof=1)
        # MAE_mean = np.mean(MAE_list)
        # MAE_std = np.std(MAE_list, ddof=1)
        # MLSD_mean = np.mean(MLSD_list)
        # MLSD_std = np.std(MLSD_list, ddof=1)
        # MR_mean = np.mean(MR_list)
        # MR_std = np.std(MR_list, ddof=1)
        # print('Test RMSE %f %f MAE %f %f MLSD %f %f MR %f %f'
        #       % (RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
        # # ==========
        # f_input = open('res/%s_MultiAggLP_rec.txt' % data_name, 'a+')
        # f_input.write('Test RMSE %f %f MAE %f %f MLSD %f %f MR %f %f Time %s\n'
        #               % (RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std, current_time))
        # f_input.close()
        # ============================================
        # ============================================
