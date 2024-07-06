# _*_ coding : utf-8 _*_
# @Time : 2024/7/5 21:16
# @Author : wfr
# @file : BL_DySAT_logistic
# @Project : IDEA

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

data_name = 'GM_2000_2'
num_nodes = 100
window_size = 10
num_snaps = 180
num_val_snaps = 10
num_test_snaps = 20
num_train_snaps = num_snaps - num_test_snaps - num_val_snaps

# ==============
edge_seq_list = np.load('data/UAV_data/%s_edge_seq.npy' % data_name, allow_pickle=True)
edge_seq_list = edge_seq_list[:180]
# 加载嵌入
data_name = 'GM_2000_2_180'
emb_list = np.load('emb_DySAT/emb_DySAT_%s_dim=128.npy' % data_name, allow_pickle=True)


# 预测下一个时间步的链路
def prepare_data_for_link_prediction(emb_list, edge_seq_list, num_nodes, window_size, start, end):
    X = []
    y = []
    for t in range(start, end):
        emb_t = emb_list[t - window_size]  # 待预测时刻前一个时刻的嵌入
        edges = edge_seq_list[t]  # 待预测时刻的边
        edge_set = set((min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if (i, j) in edge_set:
                    label = 1
                else:
                    label = 0
                X.append(np.concatenate((emb_t[i], emb_t[j])))
                y.append(label)
    return np.array(X), np.array(y)


# 获取训练数据
X_train, y_train = prepare_data_for_link_prediction(emb_list, edge_seq_list, num_nodes, window_size, window_size,
                                                    num_train_snaps)  # [10, 150)

# 获取验证数据
X_val, y_val = prepare_data_for_link_prediction(emb_list, edge_seq_list, num_nodes, window_size, num_train_snaps,
                                                num_snaps - num_test_snaps)  # [150, 160)

# 获取测试数据
X_test, y_test = prepare_data_for_link_prediction(emb_list, edge_seq_list, num_nodes, window_size,
                                                  num_snaps - num_test_snaps,
                                                  num_snaps)  # [160, 180)

# 训练逻辑回归模型
clf = LogisticRegression(random_state=0, max_iter=1000)
clf.fit(X_train, y_train)

# 预测
y_pred_prob = clf.predict_proba(X_test)[:, 1]

# 计算AUC
auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC: {auc}")

