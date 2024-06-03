# -*- encoding: utf-8 -*-
'''
@File    :   layers.py
@Time    :   2021/02/18 14:30:13
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_scatter import scatter

import copy


class StructuralAttentionLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_heads,
                 attn_drop,
                 ffd_drop,
                 residual):
        super(StructuralAttentionLayer, self).__init__()
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

    def forward(self, graph):
        # 深拷贝，防止在后续操作中修改原始数据
        graph = copy.deepcopy(graph)
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight.reshape(-1, 1)
        H, C = self.n_heads, self.out_dim  # 16, 8
        # W*X，对特征做线性变换：e.g.第一张图[18, 143]*[143, 128]=>[18, 128]=>[18,16,8]
        x = self.lin(graph.x).view(-1, H, C)  # [N, heads, out_dim]
        # attention
        # ？？？？？？？？？？？？？？？？？？？？？？？？？
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
            out = out + self.lin_residual(graph.x)
        # 最终的特征赋到图上
        graph.x = out
        return graph

    def xavier_init(self):
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)


class TemporalAttentionLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 n_heads,
                 num_time_steps,
                 attn_drop,
                 residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout 
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input
        # torch.arange(0, self.num_time_steps)生成从0到num_time_steps-1的整数序列
        # 通过 .reshape(1, -1)将其形状变为[1, num_time_steps]
        # 通过 .repeat(inputs.shape[0], 1)将该位置编码张量的行数复制成和输入张量inputs第一维相同（即 N），列数不变，得到形状为[N, num_time_steps]的位置编码张量
        # 使用 .long().to(inputs.device)将位置编码张量转换为长整型，并将其移动到和输入张量inputs相同的设备上
        position_inputs = torch.arange(0, self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
            inputs.device)
        # 每个节点加上在各个时刻（位置）对应的位置编码
        temporal_inputs = inputs + self.position_embeddings[position_inputs]  # [N, T, F] = [143,16,128]

        # 2: Query, Key based multi-head self attention.
        # torch.tensordot()的参数dims=([2], [0])表示两个张量进行点积时分别需要缩并的维度：[N, T, F]dot[F, F] => [N, T, F]
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))  # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))  # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))  # [N, T, F]

        # 3: Split, concat and scale. 在第三个维度切分为16个头，在第一个维度拼接
        split_size = int(q.shape[-1] / self.n_heads)  # 每个head的维度为128/16=8
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)  # [h*N, T, F/h] = [2288,16,8]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)  # [h*N, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)  # [h*N, T, F/h]

        # 对k_的第二、第三维度进行转置，然后q_和k_的第二、第三维度进行矩阵乘法（第一维度用来拼接）[h*N, T, F/h]dot[h*N, F/h, T] =>[h*N, T, T]
        outputs = torch.matmul(q_, k_.permute(0, 2, 1))  # [h*N, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)  # Q*K/(T)^0.5
        # 4: Masked (causal) softmax to compute attention weights.
        # 生成与outputs[0]维度相同的全1矩阵，注意outputs[0].shape=[16, 16]，而outputs.shape=[2288, 16, 16]
        diag_val = torch.ones_like(outputs[0])
        # 获得下三角矩阵（下三角全1，含对角线，其余全0）
        tril = torch.tril(diag_val)
        # 复制该下三角矩阵为h*N（2288）份
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)  # [h*N, T, T]
        # 与masks形状相同的，值全为负无穷 的矩阵
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        # 根据掩码矩阵，将outputs中等于0的位置替换为填充值
        outputs = torch.where(masks == 0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs  # [h*N, T, T]

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),
                            dim=2)  # [N, T, F]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)
