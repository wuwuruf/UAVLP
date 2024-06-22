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
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch.nn import Parameter
from torch_scatter import scatter

from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WeightedGAT(Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_heads,
                 drop_rate):
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

        self.attn_drop = nn.Dropout(drop_rate)
        self.ffd_drop = nn.Dropout(drop_rate)

        # self.residual = residual
        # if self.residual:
        #     self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)

        self.xavier_init()

    # 返回GAT处理后的特征矩阵
    def forward(self, edge_index, edge_weight, feat):
        # # 深拷贝，防止在后续操作中修改原始数据
        # graph = copy.deepcopy(graph)
        # edge_index = graph.edge_index
        edge_weight = edge_weight.reshape(-1, 1)
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
        # if self.residual:
        #     out = out + self.lin_residual(feat)
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


class ILSTM(Module):
    '''
    Class to define inductive LSTM
    '''

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(ILSTM, self).__init__()
        # ====================
        self.input_dim = input_dim  # Dimensionality of input features
        self.output_dim = output_dim  # Dimension of output features
        self.dropout_rate = dropout_rate  # Dropout rate
        # ====================
        # Initialize model parameters
        self.input_wei = Init.xavier_uniform_(
            Parameter(torch.FloatTensor(self.input_dim + self.output_dim, self.output_dim)))
        self.input_bias = Parameter(torch.zeros(self.output_dim))
        self.forget_wei = Init.xavier_uniform_(
            Parameter(torch.FloatTensor(self.input_dim + self.output_dim, self.output_dim)))
        self.forget_bias = Parameter(torch.zeros(self.output_dim))
        self.cell_wei = Init.xavier_uniform_(
            Parameter(torch.FloatTensor(self.input_dim + self.output_dim, self.output_dim)))
        self.cell_bias = Parameter(torch.zeros(self.output_dim))
        self.output_wei = Init.xavier_uniform_(
            Parameter(torch.FloatTensor(self.input_dim + self.output_dim, self.output_dim)))
        self.output_bias = Parameter(torch.zeros(self.output_dim))
        # ==========
        self.param = nn.ParameterList()
        self.param.append(self.input_wei)
        self.param.append(self.input_bias)
        self.param.append(self.forget_wei)
        self.param.append(self.forget_bias)
        self.param.append(self.cell_wei)
        self.param.append(self.cell_bias)
        self.param.append(self.output_wei)
        self.param.append(self.output_bias)
        # ==========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, pre_state, pre_cell, cur_input):
        '''
        Rewrite the forward function
        :param pre_state: previous hidden state
        :param pre_cell: previous cell state
        :param cur_input: current input
        :return: next hidden state, next cell state
        '''
        # ====================
        combined = torch.cat((cur_input, pre_state), dim=1)

        # Input gate
        input_gate = torch.sigmoid(torch.mm(combined, self.param[0]) + self.param[1])

        # Forget gate
        forget_gate = torch.sigmoid(torch.mm(combined, self.param[2]) + self.param[3])

        # Cell candidate
        cell_candidate = torch.tanh(torch.mm(combined, self.param[4]) + self.param[5])

        # Output gate
        output_gate = torch.sigmoid(torch.mm(combined, self.param[6]) + self.param[7])

        # Next cell state
        next_cell = torch.mul(forget_gate, pre_cell) + torch.mul(input_gate, cell_candidate)

        # Next hidden state
        next_state = torch.mul(output_gate, torch.tanh(next_cell))
        next_state = self.dropout_layer(next_state)

        return next_state, next_cell


class SAGPool(torch.nn.Module):
    """
    选择节点，切割edge_index
    """

    def __init__(self, in_channels, ratio=0.8, Conv=GCNConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None):
        batch = edge_index.new_zeros(x.size(0))  # 维度为[节点数]的全0的tensor
        score = self.score_layer(x, edge_index).squeeze(dim=1)  # 维度为[节点数]，代表分数
        # if score.shape != torch.Size([1]):
        #     score = score.squeeze()
        perm = topk(score, self.ratio, batch)  # 根据得分选出最高的k个节点下标，返回维度为[节点数*ratio]的tensor
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        edge_index, _ = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index


class HierarchicalPool(torch.nn.Module):
    """
    输入为一张图快照，输出为其池化表示，输出为维度 hidden_dim*2 的向量，作为图的表示
    """

    def __init__(self, feat_dim, hidden_dim, pooling_ratio, n_heads, dropout_ratio):
        """
        :param feat_dim:
        :param hidden_dim:
        :param pooling_ratio:
        :param n_heads: 表示其中使用的GAT的注意力头数
        :param dropout_ratio:
        """
        super(HierarchicalPool, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        # self.emb_dim = emb_dim

        self.pooling_ratio = pooling_ratio

        # self.conv1 = WeightedGAT(self.feat_dim, self.hidden_dim, self.n_heads, self.dropout_ratio)
        # self.pool1 = SAGPool(self.hidden_dim, ratio=self.pooling_ratio)
        # self.conv2 = WeightedGAT(self.hidden_dim, self.hidden_dim, self.n_heads, self.dropout_ratio)
        # self.pool2 = SAGPool(self.hidden_dim, ratio=self.pooling_ratio)
        # self.conv3 = WeightedGAT(self.hidden_dim, self.hidden_dim, self.n_heads, self.dropout_ratio)
        # self.pool3 = SAGPool(self.hidden_dim, ratio=self.pooling_ratio)

        self.conv1 = GCNConv(self.feat_dim, self.hidden_dim)
        self.pool1 = SAGPool(self.hidden_dim, ratio=self.pooling_ratio)
        # ==========================简化
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.pool2 = SAGPool(self.hidden_dim, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.pool3 = SAGPool(self.hidden_dim, ratio=self.pooling_ratio)

        self.dropout_layer = nn.Dropout(dropout_ratio)

    def forward(self, edge_index, edge_weight, feat):
        x = F.relu(self.conv1(feat, edge_index))
        x, edge_index = self.pool1(x, edge_index, None)  # 这里输出的edge_index与原来的edge_index边顺序是不同的
        # edge_weight_1 = [edge_weight[i] for i in range(len(edge_weight)) if
        #                  (edge_index[0][i].item(), edge_index[1][i].item()) in zip(edge_index_1[0].tolist(),
        #                                                                            edge_index_1[1].tolist())]
        # edge_weight_1 = torch.FloatTensor(edge_weight_1).to(device)
        x1 = torch.cat([gmp(x, batch=None), gap(x, batch=None)], dim=1)

        # ========================简化
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index = self.pool2(x, edge_index, None)
        # edge_weight_2 = [edge_weight_1[i] for i in range(len(edge_weight_1)) if
        #                  (edge_index_1[0][i], edge_index_1[1][i]) in zip(*edge_index_2)]
        # edge_weight_2 = torch.FloatTensor(edge_weight_2).to(device)
        x2 = torch.cat([gmp(x, batch=None), gap(x, batch=None)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index = self.pool3(x, edge_index, None)
        # edge_weight_3...
        x3 = torch.cat([gmp(x, batch=None), gap(x, batch=None)], dim=1)

        x = x1 + x2 + x3

        x = self.dropout_layer(x)

        # return x1

        return x


class AttMultiAgg(Module):
    """
    对三个尺度的表示进行注意力聚合
    （没用dropout，先看效果）
    """

    def __init__(self, input_dim, output_dim, dropout_ratio):
        """
        :param input_dim: 等于层次池化层的hidden_dim*2
        :param output_dim:
        """
        super(AttMultiAgg, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lin = nn.Linear(self.input_dim, self.output_dim, bias=False)  # W 用于对特征线性变换
        self.att_a_T = nn.Parameter(torch.Tensor(self.output_dim * 2, 1))  # a^T 是列向量
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.elu = nn.ELU()
        self.dropout_layer = nn.Dropout(dropout_ratio)
        self.xavier_init()  # nn.Parameter需要初始化

    def forward(self, micro_x, meso_x, macro_x):
        micro_x = self.lin(micro_x)
        meso_x = self.lin(meso_x)
        macro_x = self.lin(macro_x)
        # =====================
        micro_micro_x = torch.cat([micro_x, micro_x], dim=1)  # 沿着列拼接，[节点数, output_dim*2]
        micro_meso_x = torch.cat([micro_x, meso_x], dim=1)
        micro_macro_x = torch.cat([micro_x, macro_x], dim=1)
        # =====================
        micro_micro_score = self.leaky_relu(torch.matmul(micro_micro_x, self.att_a_T))  # [节点数, 1]
        micro_meso_score = self.leaky_relu(torch.matmul(micro_meso_x, self.att_a_T))
        micro_macro_score = self.leaky_relu(torch.matmul(micro_macro_x, self.att_a_T))
        # =====================
        att_weight = F.softmax(torch.cat([micro_micro_score, micro_meso_score, micro_macro_score], dim=1),
                               dim=1)  # 对每一行做softmax
        # =====================
        micro_x = att_weight[:, 0].unsqueeze(1) * micro_x
        meso_x = att_weight[:, 1].unsqueeze(1) * meso_x
        macro_x = att_weight[:, 2].unsqueeze(1) * macro_x
        # =====================
        x = micro_x + meso_x + macro_x
        x = self.elu(x)
        x = self.dropout_layer(x)

        return x

    def xavier_init(self):
        nn.init.xavier_uniform_(self.att_a_T)


class FCNN(nn.Module):
    """
    解码器
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_ratio):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # # Xavier初始化
        # init.xavier_uniform_(self.fc1.weight)
        # init.constant_(self.fc1.bias, 0)
        # init.xavier_uniform_(self.fc2.weight)
        # init.constant_(self.fc2.bias, 0)

        # He初始化
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)  # 将偏置初始化为0
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.constant_(self.fc2.bias, 0)  # 将偏置初始化为0

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
