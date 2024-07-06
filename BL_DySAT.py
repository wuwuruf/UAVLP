# BL_DySAT_emb_and_logistic.py

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
import datetime

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

torch.autograd.set_detect_anomaly(True)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


if __name__ == "__main__":

    device = torch.device('cuda')
    start_time = datetime.datetime.now().time().strftime("%H:%M:%S")

    setup_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, nargs='?', default=180,
                        help="total time steps used for train, eval and test")
    parser.add_argument('--dataset', type=str, nargs='?', default='GM_2000_4', help='dataset name')
    parser.add_argument('--num_nodes', type=int, nargs='?', default=100, help='number of nodes')
    parser.add_argument('--GPU_ID', type=int, nargs='?', default=0, help='GPU_ID (0/1 etc.)')
    parser.add_argument('--epochs', type=int, nargs='?', default=30, help='# epochs')
    parser.add_argument('--val_freq', type=int, nargs='?', default=1, help='Validation frequency (in epochs)')
    parser.add_argument('--test_freq', type=int, nargs='?', default=1, help='Testing frequency (in epochs)')
    parser.add_argument('--batch_size', type=int, nargs='?', default=512, help='Batch size (# nodes)')
    parser.add_argument('--featureless', type=bool, nargs='?', default=True, help='True if one-hot encoding.')
    parser.add_argument("--early_stop", type=int, default=10, help="patient")
    parser.add_argument('--residual', type=bool, nargs='?', default=True, help='Use residual')
    parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10, help='# negative samples per positive')
    parser.add_argument('--walk_len', type=int, nargs='?', default=20, help='Walk length for random walk sampling')
    parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0, help='Weightage for negative samples')
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                        help='Spatial (structural) attention Dropout (1 - keep probability).')
    parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.1,
                        help='Temporal attention Dropout (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
                        help='Encoder layer config: # attention heads in each GAT layer')
    parser.add_argument('--structural_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each GAT layer')
    parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                        help='Encoder layer config: # attention heads in each Temporal layer')
    parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each Temporal layer')
    parser.add_argument('--position_ffn', type=str, nargs='?', default='True', help='Position wise feedforward')
    parser.add_argument('--window', type=int, nargs='?', default=10,
                        help='Window for temporal attention (default : -1 => full)')
    args = parser.parse_args()
    print(args)

    num_nodes = 100
    window_size = 10
    num_snaps = 180
    num_val_snaps = 10
    num_test_snaps = 20
    num_train_snaps = num_snaps - num_test_snaps - num_val_snaps

    edge_seq_list = np.load('data/UAV_data/%s_edge_seq.npy' % args.dataset, allow_pickle=True)
    edge_seq_list = edge_seq_list[:180]
    max_thres = 0
    for i in range(len(edge_seq_list)):
        for j in range(len(edge_seq_list[i])):
            max_thres = max(edge_seq_list[i][j][2], max_thres)
    feat = np.load('data/UAV_data/%s_feat.npy' % args.dataset, allow_pickle=True)
    feat_list = []
    for i in range(args.time_steps):
        adj = get_adj_wei(edge_seq_list[i], args.num_nodes, max_thres)
        feat_list.append(np.concatenate((feat, adj), axis=1))
    data_name = 'GM_2000_4_180'
    graphs = []
    adjs = []
    feats = []
    for j in range(len(edge_seq_list)):
        edge_seq = edge_seq_list[j]
        G = nx.Graph()
        for i, f in enumerate(feat_list[j]):
            G.add_node(i, feature=f)
        for edge in edge_seq:
            node1, node2, weight = edge
            G.add_edge(node1, node2, weight=weight)
        graphs.append(G)
        adj = nx.adjacency_matrix(G)
        adjs.append(adj)
        feat_matrix = np.array([n[1]['feature'] for n in G.nodes(data=True)])
        feat_matrix_sparse = sp.csr_matrix(feat_matrix)
        feats.append(feat_matrix_sparse)

    context_pairs_train = get_context_pairs(graphs, adjs)
    dataset = MyDataset(args, graphs, feats, adjs, context_pairs_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
                            collate_fn=MyDataset.collate_fn)
    model = DySAT(args, feats[0].shape[1], args.time_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_auc = 0
    best_emb_list = None

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        for idx, feed_dict in enumerate(dataloader):
            for t in range(args.window - 1, args.time_steps - 1):
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
            for t in range(args.window - 1, args.time_steps - 1):
                feed_dict = to_device(feed_dict, device)
                emb = model(feed_dict['graphs'][t + 1 - args.window: t + 1])[:, -1, :].detach().cpu().numpy()
                emb_list.append(emb)

        # 验证嵌入的效果，计算AUC
        X_train, y_train = prepare_data_for_link_prediction(emb_list, edge_seq_list, args.num_nodes,
                                                            args.window,
                                                            args.window, num_train_snaps)  # [10, 150)
        X_test, y_test = prepare_data_for_link_prediction(emb_list, edge_seq_list, args.num_nodes, args.window,
                                                          args.time_steps - num_test_snaps,
                                                          args.time_steps)  # [160, 180)
        clf = LogisticRegression(random_state=0, max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)

        if auc > best_auc:
            best_auc = auc
            best_emb_list = emb_list

        print("epoch:%d, AUC:%f, best AUC:%f" % (epoch, auc, best_auc))

    # 评估最佳嵌入的效果
    # 保存最佳嵌入
    np.save('emb_DySAT/emb_DySAT_best_%s_dim=128.npy' % data_name, np.array(best_emb_list))
    X_train, y_train = prepare_data_for_link_prediction(best_emb_list, edge_seq_list, args.num_nodes,
                                                        args.window,
                                                        args.window, num_train_snaps)  # [10, 150)
    X_test, y_test = prepare_data_for_link_prediction(best_emb_list, edge_seq_list, args.num_nodes, args.window,
                                                      args.time_steps - num_test_snaps, args.time_steps)  # [160, 180)
    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred)
    print("start: %s end: %s" % (start_time, datetime.datetime.now().time().strftime("%H:%M:%S")))
    print("Best epoch's embeddings test AUC: %f" % test_auc)
