# _*_ coding : utf-8 _*_
# @Time : 2024/6/13 22:52
# @Author : wfr
# @file : MultiAggLP_demo
# @Project : IDEA

import torch
import pickle
import torch.optim as optim
import torch_geometric as tg
from my_modules.model_concat_pre_train import MultiAggLP_decoder
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

edge_seq_list = np.load('data/UAV_data/%s_edge_seq.npy' % data_name, allow_pickle=True)
edge_seq_list = edge_seq_list[0:180]
data_name = 'GM_2000_4_180'
emb_list = np.load('emb_MultiAggLP/emb_MultiAggLP_%s_dim=128.npy' % data_name, allow_pickle=True)  # 实际是128*3=384维
emb_list = [torch.FloatTensor(emb).to(device) for emb in emb_list]
# ==================
# 定义模型和优化器
model = MultiAggLP_decoder(agg_feat_dim, RNN_dims, decoder_dims, dropout_rate).to(device)
opt = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)

# ==================
best_AUC = 0.
best_val_AUC = 0.
no_improve_epochs = 0
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
    iteration_count = 0
    # =================
    for tau in indices:  # 遍历训练集的所有待预测时刻
        # ================
        # 当前窗口的预测资料
        cur_emb_list = emb_list[tau - win_size: tau]
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
        pred_adj_list = model(win_size, num_nodes, cur_emb_list, pred_flag=False)
        iteration_count += 1
        loss = get_corss_reg_loss(sparse_beta, gnd_list, pred_adj_list, theta, lambd_cross, lambd_reg)
        loss.backward()  # 累积梯度

        if iteration_count % step_interval == 0:
            opt.step()
            opt.zero_grad()
            iteration_count = 0

        # ===============
        loss_list.append(loss.item())
    # 在最后一次循环后，确保执行梯度更新
    if iteration_count % step_interval != 0:
        opt.step()
        opt.zero_grad()
    loss_mean = np.mean(loss_list)
    print('Epoch#%d Train G-Loss %f' % (epoch, loss_mean))

    if save_flag:
        torch.save(model, 'my_pt/MultiAggLP_newGAT_%d.pkl' % epoch)
    # =====================
    # 验证模型
    model.eval()
    # =============
    AUC_list = []
    f1_score_list = []
    precision_list = []
    recall_list = []
    for tau in range(num_snaps - num_test_snaps - num_val_snaps, num_snaps - num_test_snaps):  # 遍历验证集的所有待预测时刻
        # ================
        # 当前窗口的预测资料
        cur_emb_list = emb_list[tau - win_size: tau]
        # ================
        # 预测
        pred_adj_list = model(win_size, num_nodes, cur_emb_list, pred_flag=True)
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
    best_val_AUC = max(best_val_AUC, AUC_mean)
    if AUC_mean < best_val_AUC:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stop_epochs:
            break
    else:
        no_improve_epochs = 0
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
    f_input = open('res/%s_MultiAggLP_norm_weipool_lossadd_step5_lstm256_concat_pre_train_binary_rec.txt' % data_name,
                   'a+')
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
        AUC_list = []
        f1_score_list = []
        precision_list = []
        recall_list = []
        for tau in range(num_snaps - num_test_snaps, num_snaps):  # 遍历测试集的每个tau
            # ================
            # 当前窗口的预测资料
            cur_emb_list = emb_list[tau - win_size: tau]
            # ================
            # 预测
            pred_adj_list = model(win_size, num_nodes, cur_emb_list, pred_flag=True)
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
        best_AUC = max(best_AUC, AUC_mean)
        AUC_std = np.std(AUC_list, ddof=1)
        f1_score_mean = np.mean(f1_score_list)
        f1_score_std = np.std(f1_score_list, ddof=1)
        precision_mean = np.mean(precision_list)
        precision_std = np.std(precision_list, ddof=1)
        recall_mean = np.mean(recall_list)
        recall_std = np.std(recall_list, ddof=1)
        # ==============
        print('Test AUC %f %f f1_score %f %f precision %f %f recall %f %f best_AUC %f'
              % (
                  AUC_mean, AUC_std, f1_score_mean, f1_score_std, precision_mean, precision_std, recall_mean,
                  recall_std, best_AUC))
        # ==========
        f_input = open(
            'res/%s_MultiAggLP_norm_weipool_lossadd_step5_lstm256_concat_pre_train_binary_rec.txt' % data_name,
            'a+')
        f_input.write('Test AUC %f %f f1_score %f %f precision %f %f recall %f %f best_AUC %f Time %s\n'
                      % (AUC_mean, AUC_std, f1_score_mean, f1_score_std, precision_mean, precision_std, recall_mean,
                         recall_std, best_AUC, current_time))
        f_input.close()
