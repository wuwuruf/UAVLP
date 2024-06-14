# _*_ coding : utf-8 _*_
# @Time : 2024/6/14 17:33
# @Author : wfr
# @file : utils
# @Project : IDEA

import numpy as np


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
