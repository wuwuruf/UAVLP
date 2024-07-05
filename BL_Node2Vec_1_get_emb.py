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

device = torch.device('cuda')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)
data_name = 'GM_2000_2'
num_nodes = 100
train_epochs = 30
num_snaps = 180
emb_dim = 128

edge_seq_list = np.load('data/UAV_data/%s_edge_seq.npy' % data_name, allow_pickle=True)
edge_seq_list = edge_seq_list[0:180]
data_name = 'GM_2000_2_180'

# ===========================================================
# 学习除了最后一个快照的嵌入
emb_list = []
for i in range(num_snaps - 1):
    edge_seq = edge_seq_list[i]
    adj = get_adj_no_wei(edge_seq, num_nodes)
    adj_sp = sp.coo_matrix(adj, dtype=np.float32)
    edge_index, _ = tg.utils.from_scipy_sparse_matrix(adj_sp)
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

    for emb_epoch in range(train_epochs):
        model.train()
        # total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            # total_loss += loss.item()
            # total_loss = total_loss / len(loader)
        if emb_epoch == train_epochs - 1:
            print('TimeStep-%d-Emb done' % i)

    model.eval()
    emb = model()
    emb = emb.cpu().data.numpy()  # 转为numpy类型
    emb_list.append(emb)

# 保存嵌入
# 含num_snaps - 1数量的快照嵌入，不包含最后一张快照的嵌入
np.save('emb_Node2Vec/emb_Node2Vec_1_%s_dim=128.npy' % data_name, np.array(emb_list))
