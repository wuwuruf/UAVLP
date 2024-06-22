# _*_ coding : utf-8 _*_
# @Time : 2024/5/20 15:16
# @Author : wfr
# @file : BL_Node2Vec_get_emb
# @Project : IDEA
# 注意

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import Node2Vec
from modules.loss import get_decoder_loss
from utils import *
import scipy.sparse
import random
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)
data_name = 'UAV_RPGM_360_r=300'
num_nodes = 100
num_snaps = 180
emb_dim = 64

edge_seq = np.load('data/UAV_data/%s_edge_seq.npy' % data_name, allow_pickle=True)
edge_seq = edge_seq[0:180]
data_name = 'UAV_RPGM_180_r=300'
# ===========================================================
# 学习除了最后一个快照的嵌入
emb_list = []
for i in range(num_snaps - 1):
    # 去掉edge_seq中的边权重，并转为适合输入Node2Vec模型的格式
    edge_index = [[], []]
    for edge in edge_seq[i]:
        edge_index[0].append(edge[0])
        edge_index[1].append(edge[1])

    edge_index = torch.LongTensor(edge_index)

    model = Node2Vec(edge_index,
                     embedding_dim=emb_dim,
                     walk_length=40,
                     context_size=10,
                     walks_per_node=10,
                     num_negative_samples=1,
                     p=1.0,
                     q=1.0,
                     num_nodes=num_nodes,
                     sparse=True).to(device)
    loader = model.loader(batch_size=256, shuffle=True)

    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    for emb_epoch in range(1, 101):
        model.train()
        # total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            # total_loss += loss.item()
            # total_loss = total_loss / len(loader)
        if emb_epoch == 100:
            print('TimeStep-%d-Emb done' % i)

    emb = model()
    emb = emb.cpu().data.numpy()  # 转为numpy类型
    emb_list.append(emb)

# 保存嵌入
# 含num_snaps - 1数量的快照嵌入，不包含最后一张快照的嵌入
np.save('emb_Node2Vec/emb_Node2Vec_%s_dim=64.npy' % data_name, np.array(emb_list))
