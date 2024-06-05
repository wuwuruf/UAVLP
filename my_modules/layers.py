# _*_ coding : utf-8 _*_
# @Time : 2024/6/4 16:13
# @Author : wfr
# @file : layers
# @Project : IDEA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as Init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from torch_geometric.utils import softmax
from torch_scatter import scatter

import copy


class WeightedGAT(Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_heads,
                 attn_drop,
                 ffd_drop,
                 residual):
        super(WeightedGAT, self).__init__()
        # 每个头的特征维度
        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.act = nn.ELU()

        # 线性层对特征做线性变换（W*X），W维度为[143,128]
        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        # 定义α向量（将一个α向量拆成att_l和att_r）
        # 通过nn.Parameter()函数创建的PyTorch可学习参数对象，维度为(1, n_heads, self.out_dim)
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.attn_drop = nn.Dropout(attn_drop)
        self.ffd_drop = nn.Dropout(ffd_drop)

        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)

        self.xavier_init()

    # 返回GAT处理后的特征矩阵
    def forward(self, edge_index, edge_weight, feat):
        # # 深拷贝，防止在后续操作中修改原始数据
        # graph = copy.deepcopy(graph)
        # edge_index = graph.edge_index
        # edge_weight = graph.edge_weight.reshape(-1, 1)
        H, C = self.n_heads, self.out_dim  # 16, 8
        # W*X，对特征做线性变换：e.g.第一张图[18, 143]*[143, 128]=>[18, 128]=>[18,16,8]
        x = self.lin(feat).view(-1, H, C)  # [N, heads, out_dim]
        # attention
        alpha_l = (x * self.att_l).sum(dim=-1).squeeze()  # [N, heads]
        alpha_r = (x * self.att_r).sum(dim=-1).squeeze()
        alpha_l = alpha_l[edge_index[0]]  # [num_edges, heads]
        alpha_r = alpha_r[edge_index[1]]
        alpha = alpha_r + alpha_l
        # [66, 16]：66条边，16个头对应的注意力权重
        alpha = edge_weight * alpha
        alpha = self.leaky_relu(alpha)
        # 对注意力权重归一化
        coefficients = softmax(alpha, edge_index[1])  # [num_edges, heads]

        # dropout
        if self.training:
            # 对注意力权重和特征矩阵进行dropout
            coefficients = self.attn_drop(coefficients)
            x = self.ffd_drop(x)

        # 取每条边出发节点的特征（edge_index[1]为终止节点）
        x_j = x[edge_index[0]]  # [num_edges, heads, out_dim]

        # output
        # 函数：scatter(src, index, dim, reduce) 进行特征聚合
        # 原理：根据index，将index相同值对应的src元素进行定义的运算，dim为在第几维运算
        out = self.act(scatter(x_j * coefficients[:, :, None], edge_index[1], dim=0, reduce="sum"))
        # 多头的拼接
        out = out.reshape(-1, self.n_heads * self.out_dim)  # [num_nodes, output_dim]
        if self.residual:
            out = out + self.lin_residual(feat)
        # 最终的特征
        feat = out


        return feat

    def xavier_init(self):
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)


class IGRU(Module):
    '''
    Class to define inductive GRU
    '''

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(IGRU, self).__init__()
        # ====================
        self.input_dim = input_dim  # Dimensionality of input features
        self.output_dim = output_dim  # Dimension of output features
        self.dropout_rate = dropout_rate  # Dropout rate
        # ====================
        # Initialize model parameters
        self.reset_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(2 * self.input_dim, self.output_dim)))
        self.reset_bias = Parameter(torch.zeros(self.output_dim))
        self.act_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(2 * self.input_dim, self.output_dim)))
        self.act_bias = Parameter(torch.zeros(self.output_dim))
        self.update_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(2 * self.input_dim, self.output_dim)))
        self.update_bias = Parameter(torch.zeros(self.output_dim))
        # ==========
        self.param = nn.ParameterList()
        self.param.append(self.reset_wei)
        self.param.append(self.reset_bias)
        self.param.append(self.act_wei)
        self.param.append(self.act_bias)
        self.param.append(self.update_wei)
        self.param.append(self.update_bias)
        # ==========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, pre_state, cur_state):
        '''
        Rewrite the forward function
        :param pre_state: previous state
        :param cur_state: current state
        :return: next state
        '''
        # ====================
        # Reset gate
        reset_input = torch.cat((cur_state, pre_state), dim=1)
        reset_output = torch.sigmoid(torch.mm(reset_input, self.param[0]) + self.param[1])
        # ==========
        # Input activation gate
        act_input = torch.cat((cur_state, torch.mul(reset_output, pre_state)), dim=1)
        act_output = torch.tanh(torch.mm(act_input, self.param[2]) + self.param[3])
        # ==========
        # Update gate
        update_input = torch.cat((cur_state, pre_state), dim=1)
        update_output = torch.sigmoid(torch.mm(update_input, self.param[4]) + self.param[5])
        # ==========
        # Next state
        next_state = torch.mul((1 - update_output), pre_state) + torch.mul(update_output, act_output)
        next_state = self.dropout_layer(next_state)

        return next_state

