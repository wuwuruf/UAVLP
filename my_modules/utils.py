# _*_ coding : utf-8 _*_
# @Time : 2024/6/14 17:33
# @Author : wfr
# @file : utils
# @Project : IDEA

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def get_adj_wei(edges, num_nodes, max_wei):
    '''
    Function to get (dense) weighted adjacency matrix according to edge list
    :param edges: edge list
    :param num_nodes: number of nodes
    :param max_wei: maximum edge weight
    :return: adj: adjacency matrix
    '''
    adj = np.zeros((num_nodes, num_nodes))
    num_edges = len(edges)
    for i in range(num_edges):
        src = int(edges[i][0])
        dst = int(edges[i][1])
        wei = float(edges[i][2])
        if wei > max_wei:
            wei = max_wei
        adj[src, dst] = wei
        adj[dst, src] = wei
    for i in range(num_nodes):
        adj[i, i] = 0

    return adj


def get_adj_norm_wei_with_self_loop(edges, num_nodes, max_wei):
    '''
    Function to get (dense) weighted adjacency matrix according to edge list
    :param edges: edge list
    :param num_nodes: number of nodes
    :param max_wei: maximum edge weight
    :return: adj: adjacency matrix
    '''
    adj = np.zeros((num_nodes, num_nodes))
    num_edges = len(edges)
    for i in range(num_edges):
        src = int(edges[i][0])
        dst = int(edges[i][1])
        wei = float(edges[i][2])
        if wei > max_wei:
            wei = max_wei
        adj[src, dst] = wei / max_wei
        adj[dst, src] = wei / max_wei
    for i in range(num_nodes):
        adj[i, i] = 0.5  # 权重范围为(0, 1]，自环权重设置为0.5试试

    return adj


def get_adj_no_wei(edges, num_nodes):
    '''
    Function to get (dense) unweighted adjacency matrix according to edge list
    :param edges: edge list
    :param num_nodes: number of nodes
    :return: adj: adjacency matrix
    '''
    adj = np.zeros((num_nodes, num_nodes))
    num_edges = len(edges)
    for i in range(num_edges):
        src = int(edges[i][0])
        dst = int(edges[i][1])
        wei = float(edges[i][2])
        if wei > 0:
            wei = 1
        else:
            wei = 0
        adj[src, dst] = wei
        adj[dst, src] = wei
    for i in range(num_nodes):
        adj[i, i] = 0

    return adj


def get_AUC(pred_adj, gnd):
    """
    用于不加权预测
    :param pred_adj:
    :param gnd:
    :return:
    """
    return roc_auc_score(np.reshape(gnd, (-1,)), np.reshape(pred_adj, (-1,)))


def get_f1_score(pred_adj, gnd):
    """
    用于不加权预测
    :param pred_adj:
    :param gnd:
    :return:
    """
    # 将概率转换为二分类标签
    threshold = 0.5  # 设定阈值
    pred_adj = (pred_adj > threshold).astype(float)
    return f1_score(np.reshape(gnd, (-1,)), np.reshape(pred_adj, (-1,)), average='binary')


def get_precision_score(pred_adj, gnd):
    """
    用于不加权预测
    :param pred_adj:
    :param gnd:
    :return:
    """
    # 将概率转换为二分类标签
    threshold = 0.5  # 设定阈值
    pred_adj = (pred_adj > threshold).astype(float)
    return precision_score(np.reshape(gnd, (-1,)), np.reshape(pred_adj, (-1,)))


def get_recall_score(pred_adj, gnd):
    """
    用于不加权预测
    :param pred_adj:
    :param gnd:
    :return:
    """
    # 将概率转换为二分类标签
    threshold = 0.5  # 设定阈值
    pred_adj = (pred_adj > threshold).astype(float)
    return recall_score(np.reshape(gnd, (-1,)), np.reshape(pred_adj, (-1,)))


def get_D_by_edge_index_and_weight(edge_index, edge_weight, num_nodes):
    D = np.zeros((num_nodes, num_nodes))
    for i in range(len(edge_index[0])):
        node1 = edge_index[0][i]
        node2 = edge_index[1][i]
        wei = edge_weight[i]
        D[node1, node1] += wei
        D[node2, node2] += wei

    # 社团内只有一个节点的特殊情况！！
    if len(edge_index[0]) == 0:
        D[0, 0] = 1.

    return D


def get_D_by_edge_index_and_weight_tnr(edge_index, edge_weight, num_nodes):

    D = torch.zeros((num_nodes, num_nodes))
    # 使用edge_index和edge_weight更新D矩阵
    for i in range(edge_index[0].shape[0]):
        node1 = edge_index[0][i].item()
        node2 = edge_index[1][i].item()
        wei = edge_weight[i].item()
        if node1 != node2:  # 不能考虑自环
            D[node1, node1] += wei

    # 处理社团内只有一个节点的特殊情况
    if edge_index[0].shape[0] == 1:
        D[0, 0] = 1.

    return D


def get_RMSE(adj_est, gnd, num_nodes):
    '''
    Function to get the RMSE (root mean square error) metric
    :param adj_est: prediction result
    :param gnd: ground-truth
    :param num_nodes: number of nodes
    :return: RMSE metric
    '''
    # =====================
    f_norm = np.linalg.norm(gnd - adj_est, ord='fro') ** 2
    # f_norm = np.sum((gnd - adj_est)**2)
    RMSE = np.sqrt(f_norm / (num_nodes * num_nodes))

    return RMSE


def get_MAE(adj_est, gnd, num_nodes):
    '''
    Funciton to get the MAE (mean absolute error) metric
    :param adj_est: prediction result
    :param gnd: ground-truth
    :param num_nodes: number of nodes
    :return: MAE metric
    '''
    # ====================
    MAE = np.sum(np.abs(gnd - adj_est)) / (num_nodes * num_nodes)

    return MAE


def get_MLSD(adj_est, gnd, num_nodes):
    '''
    Function to get MLSD (mean logarithmic scale difference) metric
    :param adj_est: prediction result
    :param gnd: ground-truth
    :param num_nodes: number of nodes
    :return: MLSD metric
    '''
    # ====================
    epsilon = 1e-5
    adj_est_ = np.maximum(adj_est, epsilon)
    gnd_ = np.maximum(gnd, epsilon)
    MLSD = np.sum(np.abs(np.log10(adj_est_ / gnd_)))
    MLSD /= (num_nodes * num_nodes)

    return MLSD


def get_MR(adj_est, gnd, num_nodes):
    '''
    Function to get MR (mismatch rate) metric
    :param adj_est: prediction result (i.e., the estimated adjacency matrix)
    :param gnd: ground-truth
    :param num_nodes: number of nodes
    :return: MR metric
    '''
    # ====================
    mis_sum = 0
    for r in range(num_nodes):
        for c in range(num_nodes):
            if (adj_est[r, c] > 0 and gnd[r, c] == 0) or (adj_est[r, c] == 0 and gnd[r, c] > 0):
                mis_sum += 1
    # ==========
    MR = mis_sum / (num_nodes * num_nodes)

    return MR
