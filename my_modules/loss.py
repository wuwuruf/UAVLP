# _*_ coding : utf-8 _*_
# @Time : 2024/6/14 10:07
# @Author : wfr
# @file : loss
# @Project : IDEA
import torch
import torch.nn.functional as F


def get_refined_loss(beta, gnd_list, pred_adj_list, theta):
    """
    获取适应稀疏情况的不加权重构误差
    :param beta:
    :param gnd_list:
    :param pred_adj_list:
    :param theta:
    :return:
    """
    loss = 0.0
    win_size = len(pred_adj_list)
    for i in range(win_size):
        gnd = gnd_list[i]
        pred_adj = pred_adj_list[i]
        decay = (1 - theta) ** (win_size - i - 1)  # Decaying factor
        weight = gnd * (beta - 1) + 1
        loss += decay * torch.mean(torch.sum(weight * torch.square(gnd - pred_adj), dim=1), dim=-1)
    return loss


def get_corss_reg_loss(beta, gnd_list, pred_adj_list, theta):
    """
    获取适应稀疏情况的不加权重构误差，交叉熵损失和1范数正则化
    """
    loss = 0.0
    win_size = len(pred_adj_list)
    for i in range(win_size):
        gnd = gnd_list[i]
        pred_adj = pred_adj_list[i]
        decay = (1 - theta) ** (win_size - i - 1)  # Decaying factor

        # 计算原始的加权重构误差
        weight = gnd * (beta - 1) + 1
        reconstruction_loss = torch.mean(torch.sum(weight * torch.square(gnd - pred_adj), dim=1), dim=-1)

        # 计算交叉熵损失
        lambda_cross_entropy = 20
        cross_entropy_loss = lambda_cross_entropy * F.binary_cross_entropy(pred_adj, gnd)

        # 计算1范数正则化项
        lambda_reg = 0.0005
        l1_regularization = lambda_reg * torch.sum(torch.abs(pred_adj))

        # 总损失
        total_loss = decay * (reconstruction_loss + cross_entropy_loss + l1_regularization)

        loss += total_loss
    return loss


def get_single_refined_loss(beta, gnd, pred_adj):
    """
    获取适应稀疏情况的不加权重构误差
    :param beta:
    :param gnd:
    :param pred_adj:
    :return:
    """
    weight = gnd * (beta - 1) + 1
    loss = torch.mean(torch.sum(weight * torch.square(gnd - pred_adj), dim=1), dim=-1)
    return loss


def get_single_corss_reg_loss(beta, gnd, pred_adj):
    """
    获取适应稀疏情况的不加权重构误差，交叉熵损失和1范数正则化
    :param beta: 加权系数
    :param gnd: 真实邻接矩阵
    :param pred_adj: 模型预测的邻接矩阵
    :return: 总损失
    """
    # 计算原始的加权重构误差
    weight = gnd * (beta - 1) + 1
    reconstruction_loss = torch.mean(torch.sum(weight * torch.square(gnd - pred_adj), dim=1), dim=-1)

    # 计算交叉熵损失
    lambda_cross_entropy = 20
    cross_entropy_loss = lambda_cross_entropy * F.binary_cross_entropy(pred_adj, gnd)

    # 计算1范数正则化项
    lambda_reg = 0.0005
    l1_regularization = lambda_reg * torch.sum(torch.abs(pred_adj))

    # 总损失
    total_loss = reconstruction_loss + cross_entropy_loss + l1_regularization

    return total_loss


def get_loss(adj_est_list, gnd_list, max_thres, alpha, beta, theta):
    '''
    Function to define the loss of generator (in the formal optimization)
    :param adj_est_list: list of prediction results
    :param gnd_list: list of ground-truth
    :param alpha: parameter to control ME loss
    :param beta: parameter to control SDM loss
    :param theta: parameter of decaying factor
    :return: loss of generator
    '''
    # ====================
    loss = 0.0
    win_size = len(adj_est_list)
    for i in range(win_size):
        adj_est = adj_est_list[i]
        gnd = gnd_list[i]
        decay = (1 - theta) ** (win_size - i - 1)  # Decaying factor
        # ==========
        # EM (error minimization) loss
        loss += decay * alpha * torch.norm((adj_est - gnd), p='fro') ** 2
        loss += decay * alpha * torch.sum(torch.abs(adj_est - gnd))
        # SDM (scale difference minimization) loss 暂不需要MLSD ？
        epsilon = 1e-5 / max_thres
        E = epsilon * torch.ones_like(adj_est)
        q = adj_est
        q = torch.where(q < epsilon, E, q)
        p = gnd
        p = torch.where(p < epsilon, E, p)
        loss += decay * beta * torch.sum(torch.abs(torch.log10(p / q)))

    return loss


def get_single_loss(adj_est, gnd, max_thres, alpha, beta):
    # ====================
    loss = 0.0
    # ==========
    # EM (error minimization) loss
    loss += alpha * torch.norm((adj_est - gnd), p='fro') ** 2
    loss += alpha * torch.sum(torch.abs(adj_est - gnd))
    # SDM (scale difference minimization) loss 暂不需要MLSD
    epsilon = 1e-5 / max_thres
    E = epsilon * torch.ones_like(adj_est)
    q = adj_est
    q = torch.where(q < epsilon, E, q)
    p = gnd
    p = torch.where(p < epsilon, E, p)
    loss += beta * torch.sum(torch.abs(torch.log10(p / q)))

    return loss
