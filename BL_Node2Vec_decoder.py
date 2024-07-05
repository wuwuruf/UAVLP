# _*_ coding : utf-8 _*_
# @Time : 2024/5/18 15:33
# @Author : wfr
# @file : BL_Node2Vec
# @Project : IDEA
# Node2Vec, 用Gt-1预测Gt

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import Node2Vec
from my_modules.loss import *
from my_modules.utils import *
from utils import *
import scipy.sparse
import random
import datetime

device = torch.device('cuda')


class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_nodes)

        # # Xavier初始化
        # init.xavier_uniform_(self.fc1.weight)
        # init.constant_(self.fc1.bias, 0)
        # init.xavier_uniform_(self.fc2.weight)
        # init.constant_(self.fc2.bias, 0)

        # He初始化
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)  # 将偏置初始化为0
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.constant_(self.fc2.bias, 0)  # 将偏置初始化为0

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)
data_name = 'GM_2000_2'
num_nodes = 100
num_snaps = 180
num_val_snaps = 10
num_test_snaps = 20  # Number of test snapshots
num_train_snaps = num_snaps - num_test_snaps - num_val_snaps
num_epochs = 800
emb_dim = 128
decoder_hidden_dim = 256

# =================================================
sparse_beta = 10
lambd_cross = 5
lambd_reg = 0.001
step_interval = 5
early_stop_epochs = 50
# =================================================
epsilon = 0.01  # Threshold of the zero-refining

edge_seq = np.load('data/UAV_data/%s_edge_seq.npy' % data_name, allow_pickle=True)
data_name = 'GM_2000_2_180'
emb_list = np.load('emb_Node2Vec/emb_Node2Vec_1_%s_dim=%d.npy' % (data_name, emb_dim))
edge_seq = edge_seq[0:180]
# ===============================================
decoder = FCNN(emb_dim, decoder_hidden_dim, num_nodes).to(device)
decoder_opt = optim.Adam(decoder.parameters(), lr=5e-4, weight_decay=1e-5)
# ===============================================
# 使用训练集训练用于预测的解码器（两层MLP）
best_AUC = 0.
best_val_AUC = 0.
no_improve_epochs = 0
for epoch in range(num_epochs):
    current_time = datetime.datetime.now().time().strftime("%H:%M:%S")
    decoder_loss_list = []
    decoder.train()
    # =======================
    # 每个epoch的打乱顺序都不同
    random.seed(epoch)
    # 生成从win_size到num_train_snaps范围内的列表
    indices = list(range(num_train_snaps))
    # 对列表进行随机打乱
    random.shuffle(indices)
    # =======================
    iteration_count = 0
    for i in indices:
        emb_tnr = torch.FloatTensor(emb_list[i]).to(device)  # 获取当前时刻的嵌入
        emb_tnr = F.normalize(emb_tnr, dim=0, p=2)  # 对嵌入矩阵列向量进行标准化
        adj_est = decoder(emb_tnr)
        # =================================
        # gnd = get_adj_wei(edge_seq[i + 1], num_nodes, max_thres)  # 注意是预测下一时刻，要i+1
        # gnd_norm = gnd / max_thres
        # gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
        # # 计算损失用的是[0, 1]之间的矩阵，计算指标用的是是真实值矩阵
        # decoder_loss = get_decoder_loss(adj_est, gnd_tnr, max_thres, alpha, beta)
        # =================================
        gnd = get_adj_no_wei(edge_seq[i + 1], num_nodes)
        gnd_tnr = torch.FloatTensor(gnd).to(device)
        decoder_loss = get_single_corss_reg_loss(sparse_beta, gnd_tnr, adj_est, lambd_cross, lambd_reg)
        # ==========================
        decoder_loss.backward()  # 累积梯度
        if iteration_count % step_interval == 0:
            decoder_opt.step()
            decoder_opt.zero_grad()
            iteration_count = 0
        decoder_loss_list.append(decoder_loss.item())
    # 在最后一次循环后，确保执行梯度更新
    if iteration_count % step_interval != 0:
        decoder_opt.step()
        decoder_opt.zero_grad()
    decoder_loss_mean = np.mean(decoder_loss_list)
    # print("Epoch-%d decoder_loss:" % epoch, decoder_loss_mean)

    # ===================================================
    # 使用验证集进行验证
    decoder.eval()
    # RMSE_list = []
    # MAE_list = []
    # MLSD_list = []
    # MR_list = []
    AUC_list = []
    f1_score_list = []
    for i in range(num_snaps - num_test_snaps - num_val_snaps, num_snaps - num_test_snaps):
        # Get and refine the prediction result
        emb_tnr = torch.FloatTensor(emb_list[i]).to(device)
        adj_est = decoder(emb_tnr)
        adj_est = adj_est.cpu().data.numpy()  # 张量转为numpy类型
        # ========================================
        # 不加权二分类预测
        # ==============
        # 获取真实的不加权0-1邻接矩阵
        gnd = get_adj_no_wei(edge_seq[i + 1], num_nodes)
        # ===============
        # 计算评价指标
        AUC = get_AUC(adj_est, gnd)
        f1_score = get_f1_score(adj_est, gnd)
        # ===============
        AUC_list.append(AUC)
        f1_score_list.append(f1_score)
        # ==============
    AUC_mean = np.mean(AUC_list)
    best_val_AUC = max(best_val_AUC, AUC_mean)
    if AUC_mean < best_val_AUC:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stop_epochs:
            break
    else:
        no_improve_epochs = 0
    # ==============
    AUC_mean = np.mean(AUC_list)
    AUC_std = np.std(AUC_list, ddof=1)
    f1_score_mean = np.mean(f1_score_list)
    f1_score_std = np.std(f1_score_list, ddof=1)
    # ==============
    print('Val #%d AUC %f %f f1_score %f %f'
          % (epoch, AUC_mean, AUC_std, f1_score_mean, f1_score_std))
    # ==========
    f_input = open('res/%s_Node2Vec_binary_rec.txt' % data_name, 'a+')
    f_input.write('Val #%d AUC %f %f f1_score %f %f Time %s\n'
                  % (epoch, AUC_mean, AUC_std, f1_score_mean, f1_score_std, current_time))
    f_input.close()
    # =====================================
    #     adj_est *= max_thres
    #     for r in range(num_nodes):
    #         adj_est[r, r] = 0
    #     for r in range(num_nodes):
    #         for c in range(num_nodes):
    #             if adj_est[r, c] <= epsilon:
    #                 adj_est[r, c] = 0
    #
    #     # Get the ground-truth
    #     gnd = get_adj_wei(edge_seq[i + 1], num_nodes, max_thres)
    #
    #     # Evaluate the prediction result
    #     RMSE = get_RMSE(adj_est, gnd, num_nodes)
    #     MAE = get_MAE(adj_est, gnd, num_nodes)
    #     MLSD = get_MLSD(adj_est, gnd, num_nodes)
    #     MR = get_MR(adj_est, gnd, num_nodes)
    #
    #     RMSE_list.append(RMSE)
    #     MAE_list.append(MAE)
    #     MLSD_list.append(MLSD)
    #     MR_list.append(MR)
    #
    # RMSE_mean = np.mean(RMSE_list)
    # RMSE_std = np.std(RMSE_list, ddof=1)
    # MAE_mean = np.mean(MAE_list)
    # MAE_std = np.std(MAE_list, ddof=1)
    # MLSD_mean = np.mean(MLSD_list)
    # MLSD_std = np.std(MLSD_list, ddof=1)
    # MR_mean = np.mean(MR_list)
    # MR_std = np.std(MR_list, ddof=1)
    # print('Val #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f\n'
    #       % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
    # f_input = open('res/%s_Node2Vec_rec.txt' % data_name, 'a+')
    # f_input.write('Val #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f Time %s\n'
    #               % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std, current_time))
    # f_input.close()
    if (epoch + 1) % 5 == 0:
        # ======================================================================
        # 使用解码器对测试集进行预测
        decoder.eval()
        # RMSE_list = []
        # MAE_list = []
        # MLSD_list = []
        # MR_list = []
        AUC_list = []
        f1_score_list = []
        current_time = datetime.datetime.now().time().strftime("%H:%M:%S")
        count = 0
        for i in range(num_snaps - num_test_snaps, num_snaps - 1):
            count = count + 1
            # Get and refine the prediction result
            emb_tnr = torch.FloatTensor(emb_list[i]).to(device)
            emb_tnr = F.normalize(emb_tnr, dim=0, p=2)
            adj_est = decoder(emb_tnr)
            adj_est = adj_est.cpu().data.numpy()  # 张量转为numpy类型
            # ========================================
            # 不加权二分类预测
            # ==============
            # 获取真实的不加权0-1邻接矩阵
            gnd = get_adj_no_wei(edge_seq[i + 1], num_nodes)
            # ===============
            # 计算评价指标
            AUC = get_AUC(adj_est, gnd)
            f1_score = get_f1_score(adj_est, gnd)
            # ===============
            AUC_list.append(AUC)
            f1_score_list.append(f1_score)
        # ==============
        AUC_mean = np.mean(AUC_list)
        best_AUC = max(best_AUC, AUC_mean)
        AUC_std = np.std(AUC_list, ddof=1)
        f1_score_mean = np.mean(f1_score_list)
        f1_score_std = np.std(f1_score_list, ddof=1)
        # ==============
        print('Test AUC %f %f f1_score %f %f best_AUC %f'
              % (AUC_mean, AUC_std, f1_score_mean, f1_score_std, best_AUC))
        # ==========
        f_input = open('res/%s_Node2Vec_binary_rec.txt' % data_name, 'a+')
        f_input.write('Test AUC %f %f f1_score %f %f best_AUC %f Time %s\n'
                      % (AUC_mean, AUC_std, f1_score_mean, f1_score_std, best_AUC, current_time))
        f_input.close()
    # =====================================
    #     adj_est *= max_thres
    #     for r in range(num_nodes):
    #         adj_est[r, r] = 0
    #     for r in range(num_nodes):
    #         for c in range(num_nodes):
    #             if adj_est[r, c] <= epsilon:
    #                 adj_est[r, c] = 0
    #
    #     # Get the ground-truth
    #     gnd = get_adj_wei(edge_seq[i + 1], num_nodes, max_thres)
    #
    #     # Evaluate the prediction result
    #     RMSE = get_RMSE(adj_est, gnd, num_nodes)
    #     MAE = get_MAE(adj_est, gnd, num_nodes)
    #     MLSD = get_MLSD(adj_est, gnd, num_nodes)
    #     MR = get_MR(adj_est, gnd, num_nodes)
    #
    #     RMSE_list.append(RMSE)
    #     MAE_list.append(MAE)
    #     MLSD_list.append(MLSD)
    #     MR_list.append(MR)
    # print("test_num:%d" % count)
    # # ====================
    # RMSE_mean = np.mean(RMSE_list)
    # RMSE_std = np.std(RMSE_list, ddof=1)
    # MAE_mean = np.mean(MAE_list)
    # MAE_std = np.std(MAE_list, ddof=1)
    # MLSD_mean = np.mean(MLSD_list)
    # MLSD_std = np.std(MLSD_list, ddof=1)
    # MR_mean = np.mean(MR_list)
    # MR_std = np.std(MR_list, ddof=1)
    # print('Test RMSE %f %f MAE %f %f MLSD %f %f MR %f %f\n'
    #       % (RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
    # f_input = open('res/%s_Node2Vec_rec.txt' % data_name, 'a+')
    # f_input.write('Test RMSE %f %f MAE %f %f MLSD %f %f MR %f %f Time %s\n'
    #               % (RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std, current_time))
    # f_input.close()
