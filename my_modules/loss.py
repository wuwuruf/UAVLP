# _*_ coding : utf-8 _*_
# @Time : 2024/6/14 10:07
# @Author : wfr
# @file : loss
# @Project : IDEA
import torch
import torch.nn.functional as F

device = torch.device('cuda')


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


def get_corss_reg_loss(beta, gnd_list, pred_adj_list, theta, lambda_cross_entropy=20, lambda_reg=0.0005):
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

        # 计算交叉熵损失（pred_adj_list[3]有问题）
        cross_entropy_loss = lambda_cross_entropy * F.binary_cross_entropy(pred_adj, gnd)

        # 计算1范数正则化项
        l1_regularization = lambda_reg * torch.sum(torch.abs(pred_adj))

        # 总损失
        total_loss = decay * (reconstruction_loss + cross_entropy_loss + l1_regularization)

        loss += total_loss
    return loss


def get_emb_loss(node_embeddings, adjacency_matrix, lambda2=0.1, lambda3=0.01):
    # 一阶邻居
    A = adjacency_matrix
    # 二阶邻居
    A2 = torch.matmul(A, A)
    A2[A2 > 0] = 1  # 将非零元素置为1
    A2 = A2 - torch.diag(torch.diag(A2))  # 去掉对角线元素
    A2 = A2 - A  # 去掉一阶邻居的影响

    # 三阶邻居
    A3 = torch.matmul(A2, A)
    A3[A3 > 0] = 1  # 将非零元素置为1
    A3 = A3 - torch.diag(torch.diag(A3))  # 去掉对角线元素
    A3 = A3 - A  # 去掉一阶邻居的影响
    A3 = A3 - A2  # 去掉二阶邻居的影响

    # 计算一阶邻居的损失
    pos_loss_1 = torch.sum(A * torch.sum((node_embeddings.unsqueeze(0) - node_embeddings.unsqueeze(1)) ** 2, dim=-1))

    # 计算二阶邻居的损失
    pos_loss_2 = torch.sum(A2 * torch.sum((node_embeddings.unsqueeze(0) - node_embeddings.unsqueeze(1)) ** 2, dim=-1))

    # 计算三阶邻居的损失
    pos_loss_3 = torch.sum(A3 * torch.sum((node_embeddings.unsqueeze(0) - node_embeddings.unsqueeze(1)) ** 2, dim=-1))

    # 负采样，确保不相连节点的表示具有较大的差异
    num_nodes = adjacency_matrix.shape[0]
    neg_samples = torch.randint(0, num_nodes, (num_nodes, num_nodes))
    neg_samples = (neg_samples != torch.arange(num_nodes).unsqueeze(1)).float().to(device)

    neg_loss = torch.sum(
        neg_samples * torch.sum((node_embeddings.unsqueeze(0) - node_embeddings.unsqueeze(1)) ** 2, dim=-1))

    # 总损失
    loss = pos_loss_1 + lambda2 * pos_loss_2 + lambda3 * pos_loss_3 + neg_loss

    return loss


def get_wei_emb_loss(node_embeddings, adjacency_matrix_wei):
    # 一阶邻居
    A = adjacency_matrix_wei

    # 计算一阶邻居的损失
    pos_loss_1 = torch.sum(A * torch.sum((node_embeddings.unsqueeze(0) - node_embeddings.unsqueeze(1)) ** 2, dim=-1))

    # 负采样，确保不相连节点的表示具有较大的差异
    num_nodes = adjacency_matrix_wei.shape[0]
    neg_samples = torch.randint(0, num_nodes, (num_nodes, num_nodes))
    neg_samples = (neg_samples != torch.arange(num_nodes).unsqueeze(1)).float().to(device)

    neg_loss = torch.sum(
        neg_samples * torch.sum((node_embeddings.unsqueeze(0) - node_embeddings.unsqueeze(1)) ** 2, dim=-1))

    # 总损失
    loss = pos_loss_1 + neg_loss

    return loss


def get_corss_reg_rep_loss(beta, gnd_list, pred_adj_list, node_embeddings_list, theta, lambda_cross_entropy=20,
                           lambda_reg=0.0005,
                           lambda_rep_diff=0.005):
    """
    获取适应稀疏情况的不加权重构误差，交叉熵损失和1范数正则化
    """
    loss = 0.0
    win_size = len(pred_adj_list)
    for i in range(win_size):
        gnd = gnd_list[i]
        pred_adj = pred_adj_list[i]
        node_embeddings = node_embeddings_list[i]
        decay = (1 - theta) ** (win_size - i - 1)  # Decaying factor

        # 计算原始的加权重构误差
        weight = gnd * (beta - 1) + 1
        reconstruction_loss = torch.mean(torch.sum(weight * torch.square(gnd - pred_adj), dim=1), dim=-1)

        # 计算交叉熵损失
        cross_entropy_loss = lambda_cross_entropy * F.binary_cross_entropy(pred_adj, gnd)

        # 计算1范数正则化项
        l1_regularization = lambda_reg * torch.sum(torch.abs(pred_adj))

        # 计算有连接节点之间的表示差异误差
        node_emb_norm = torch.norm(node_embeddings, dim=1, keepdim=True)
        normalized_emb = node_embeddings / node_emb_norm
        sim_matrix = torch.mm(normalized_emb, normalized_emb.t())
        rep_diff_loss = lambda_rep_diff * torch.sum((1 - sim_matrix) * gnd)

        # 总损失
        total_loss = decay * (reconstruction_loss + cross_entropy_loss + l1_regularization + rep_diff_loss)

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


def get_single_corss_reg_loss(beta, gnd, pred_adj, lambda_cross_entropy=20, lambda_reg=0.0005):
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
    cross_entropy_loss = lambda_cross_entropy * F.binary_cross_entropy(pred_adj, gnd)

    # 计算1范数正则化项
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


def get_wei_loss(adj_est_list, gnd_list, beta, theta):
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
        pred_adj = adj_est_list[i]
        gnd = gnd_list[i]
        decay = (1 - theta) ** (win_size - i - 1)  # Decaying factor
        # ==========
        weight = gnd * (beta - 1) + 1
        loss += decay * torch.mean(torch.sum(weight * torch.square(gnd - pred_adj), dim=1), dim=-1)

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
