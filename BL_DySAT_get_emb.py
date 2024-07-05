# _*_ coding : utf-8 _*_
# @Time : 2024/5/26 13:31
# @Author : wfr
# @file : BL_DySAT_emb
# @Project : IDEA

import argparse
import networkx as nx
import numpy as np
import random
import dill
import pickle as pkl
import scipy
import scipy.sparse as sp
from torch.utils.data import DataLoader

from utils_DySAT.preprocess import load_graphs, get_context_pairs, get_evaluation_data
from utils_DySAT.minibatch import MyDataset
from utils_DySAT.utilities import to_device
from utils_DySAT.model import DySAT
from my_modules.utils import *

import torch

# 用于在进行反向传播时检测梯度计算过程中的异常
torch.autograd.set_detect_anomaly(True)


# 返回一个新的多重图对象，该图包含了t+1时刻的节点和t时刻的边
def inductive_graph(graph_former, graph_later):
    """Create the adj_train so that it includes nodes from (t+1)
       but only edges from t: this is for the purpose of inductive testing.

    Args:
        graph_former ([type]): [description]
        graph_later ([type]): [description]
    """
    newG = nx.MultiGraph()
    newG.add_nodes_from(graph_later.nodes(data=True))
    newG.add_edges_from(graph_former.edges(data=False))
    return newG


if __name__ == "__main__":

    device = torch.device('cuda')


    def setup_seed(seed):
        torch.manual_seed(seed)  # 设置PyTorch随机种子
        torch.cuda.manual_seed_all(seed)  # 设置PyTorch的CUDA随机种子（如果GPU可用）
        np.random.seed(seed)  # 设置NumPy的随机种子
        random.seed(seed)  # 设置Python内置random库的随机种子
        torch.backends.cudnn.deterministic = True  # 设置使用CUDA加速时保证结果一致性


    setup_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, nargs='?', default=180,
                        help="total time steps used for train, eval and test")
    # Experimental settings.
    parser.add_argument('--dataset', type=str, nargs='?', default='GM_2000_6',
                        help='dataset name')
    parser.add_argument('--num_nodes', type=int, nargs='?', default=100,
                        help='number of nodes')
    parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)')
    parser.add_argument('--epochs', type=int, nargs='?', default=30,
                        help='# epochs')
    parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                        help='Validation frequency (in epochs)')
    parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                        help='Testing frequency (in epochs)')
    parser.add_argument('--batch_size', type=int, nargs='?', default=512,
                        help='Batch size (# nodes)')
    parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                        help='True if one-hot encoding.')
    parser.add_argument("--early_stop", type=int, default=10,
                        help="patient")
    # 1-hot encoding is input as a sparse matrix - hence no scalability issue for large datasets.
    # Tunable hyper-params
    # TODO: Implementation has not been verified, performance may not be good.
    parser.add_argument('--residual', type=bool, nargs='?', default=True,
                        help='Use residual')
    # Number of negative samples per positive pair.
    parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
                        help='# negative samples per positive')
    # Walk length for random walk sampling.
    parser.add_argument('--walk_len', type=int, nargs='?', default=20,
                        help='Walk length for random walk sampling')
    # Weight for negative samples in the binary cross-entropy loss function.
    parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
                        help='Weightage for negative samples')
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                        help='Spatial (structural) attention Dropout (1 - keep probability).')
    parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.1,  # 0.5
                        help='Temporal attention Dropout (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                        help='Initial learning rate for self-attention model.')
    # Architecture params
    parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
                        help='Encoder layer config: # attention heads in each GAT layer')
    parser.add_argument('--structural_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each GAT layer')
    parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                        help='Encoder layer config: # attention heads in each Temporal layer')
    parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each Temporal layer')
    parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                        help='Position wise feedforward')
    parser.add_argument('--window', type=int, nargs='?', default=10,
                        help='Window for temporal attention (default : -1 => full)')
    args = parser.parse_args()
    print(args)

    # ============================================
    # 构建graphs, adjs, feats列表
    edge_seq_list = np.load('data/UAV_data/%s_edge_seq.npy' % args.dataset, allow_pickle=True)
    edge_seq_list = edge_seq_list[:180]
    max_thres = 0  # Threshold for maximum edge weight
    for i in range(len(edge_seq_list)):
        for j in range(len(edge_seq_list[i])):
            max_thres = max(edge_seq_list[i][j][2], max_thres)
    feat = np.load('data/UAV_data/%s_feat.npy' % args.dataset, allow_pickle=True)
    feat_list = []
    for i in range(args.time_steps):
        adj = get_adj_wei(edge_seq_list[i], args.num_nodes, max_thres)
        feat_list.append(np.concatenate((feat, adj), axis=1))
    data_name = 'GM_2000_6_180'
    graphs = []
    adjs = []
    feats = []
    for j in range(len(edge_seq_list)):
        edge_seq = edge_seq_list[j]
        # 创建一个新的无向图
        G = nx.Graph()
        # 添加节点特征
        for i, f in enumerate(feat_list[j]):
            G.add_node(i, feature=f)
        # 添加边和权重
        for edge in edge_seq:
            node1, node2, weight = edge
            G.add_edge(node1, node2, weight=weight)
        # 将图添加到图列表中
        graphs.append(G)
        # 生成邻接矩阵
        adj = nx.adjacency_matrix(G)
        adjs.append(adj)
        # 生成特征矩阵
        feat_matrix = np.array([n[1]['feature'] for n in G.nodes(data=True)])
        feat_matrix_sparse = sp.csr_matrix(feat_matrix)  # 转换为稀疏矩阵
        feats.append(feat_matrix_sparse)
    # ==========================================
    # node2vec的训练语料，传入图列表和邻接矩阵列表
    # 进行随机游走获取上下文节点
    # 返回值是上下文节点对列表，列表包含16个元素，对应16张图快照，列表中的元素是字典，字典的键为中心节点
    context_pairs_train = get_context_pairs(graphs, adjs)

    # build dataloader and model
    # 加载数据集，实例化该自定义类
    dataset = MyDataset(args, graphs, feats, adjs, context_pairs_train)
    # batch_size是512，节点数143小于512，因此会一次导入所有节点信息
    # collate_fn是一个可选参数，如果指定了该参数，DataLoader在获取到batch_size长度的样本列表后会调用（传入）collate_fn函数，将多个单独的样本组合成一个所需要形式的mini-batch
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=1,
                            collate_fn=MyDataset.collate_fn)
    # dataloader = NodeMinibatchIterator(args, graphs, feats, adjs, context_pairs_train, device)
    model = DySAT(args, feats[0].shape[1], args.time_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # in training
    for epoch in range(args.epochs):
        model.train()  # 将模型设置为训练模式
        epoch_loss = []
        for idx, feed_dict in enumerate(dataloader):  # 实际上由于batch_size=512，只有一个feed_dict，其中包含了所有快照的信息
            for t in range(args.window - 1, args.time_steps - 1):  # t为窗口内的最后一个时刻的下标
                feed_dict_window = {'node_1': feed_dict['node_1'][t + 1 - args.window: t + 1],
                                    'node_2': feed_dict['node_2'][t + 1 - args.window: t + 1],
                                    'node_2_neg': feed_dict['node_2_neg'][t + 1 - args.window: t + 1],
                                    'graphs': feed_dict['graphs'][t + 1 - args.window: t + 1]}
                feed_dict_window = to_device(feed_dict_window, device)
                opt.zero_grad()
                loss = model.get_loss(feed_dict_window)
                loss.backward()
                opt.step()
                epoch_loss.append(loss.item())
        epoch_loss_mean = np.mean(epoch_loss)
        print("epoch:%d, loss:%f" % (epoch, epoch_loss_mean))

    model.eval()
    emb_list = []
    for idx, feed_dict in enumerate(dataloader):
        for t in range(args.window - 1, args.time_steps - 1):  # t为窗口内的最后一个时刻的下标
            # 使用窗口内最后一个时间步的嵌入
            feed_dict = to_device(feed_dict, device)
            emb = model(feed_dict['graphs'][t + 1 - args.window: t + 1])[:, -1, :].detach().cpu().numpy()
            emb_list.append(emb)

    # 只包含了下标为[window - 1, time_steps - 2]的嵌入，用来预测下标为[window, time_steps - 1]的链路，嵌入矩阵数量为num_snaps - win_size
    np.save('emb_DySAT/emb_DySAT_%s_dim=128.npy' % data_name, np.array(emb_list))
