# _*_ coding : utf-8 _*_
# @Time : 2024/6/14 10:07
# @Author : wfr
# @file : loss
# @Project : IDEA
import torch


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
