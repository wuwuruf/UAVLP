# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/02/19 21:10:00
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss

from utils_DySAT.layers import StructuralAttentionLayer, TemporalAttentionLayer
# from utils.utilities import fixed_unigram_candidate_sampler


class DySAT(nn.Module):
    def __init__(self, args, num_features, time_length):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(DySAT, self).__init__()
        self.args = args
        # windows = -1表示full注意力
        # if args.window < 0:
        #     self.num_time_steps = time_length
        # else:
        #     self.num_time_steps = min(time_length, args.window + 1)  # window = 0 => only self.

        self.num_features = num_features
        self.num_time_steps = args.window
        # 结构多头信息[16,8,8]
        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))
        # 结构layer层信息[128]
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))
        # 时序多头信息[16]
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        # 时序layer层信息[128]
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))
        # 定义dropout
        self.spatial_drop = args.spatial_drop

        self.temporal_drop = args.temporal_drop
        # 定义model
        self.structural_attn, self.temporal_attn = self.build_model()
        # 定义损失函数(sigmoid和crossEntropy结合)
        self.bceloss = BCEWithLogitsLoss()

    def forward(self, graphs):

        # Structural Attention forward
        structural_out = []
        # 遍历时间步，对每一个时间步的图的节点进行GAT操作
        for t in range(0, self.num_time_steps):
            structural_out.append(self.structural_attn(graphs[t]))
        structural_outputs = [g.x[:, None, :] for g in structural_out]  # 矩阵的列表 list of [Ni, 1, F], Ni为第i个时间步的节点数

        # padding outputs along with Ni
        maximum_node_num = structural_outputs[-1].shape[0]  # 最大节点数
        out_dim = structural_outputs[-1].shape[-1]  # 输出特征的维度
        structural_outputs_padded = []
        for out in structural_outputs:  # 对节点进行补0，使其为同一个维度
            zero_padding = torch.zeros(maximum_node_num - out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_padded.append(padded)

        structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1)  # 按时间维度拼接起来得到形状[N, T, F]

        # Temporal Attention forward
        temporal_out = self.temporal_attn(structural_outputs_padded)

        return temporal_out

    def build_model(self):
        # 输入维度为初始特征维度
        input_dim = self.num_features

        # 1: 定义Structural Attention Layers
        # 创建一个空的神经网络序列容器，nn.Sequential() 是PyTorch中的一个模型容器，可以按照添加顺序依次执行其中包含的各个模块
        structural_attention_layers = nn.Sequential()
        # 遍历创建每一层，此处len(self.structural_layer_config)=1，只有一层
        for i in range(len(self.structural_layer_config)):
            # 设置相应参数，创建结构注意力层
            layer = StructuralAttentionLayer(input_dim=input_dim,
                                             output_dim=self.structural_layer_config[i],
                                             n_heads=self.structural_head_config[i],
                                             attn_drop=self.spatial_drop,
                                             ffd_drop=self.spatial_drop,
                                             residual=self.args.residual)
            # 每个结构注意力层都会被添加到structural_attention_layers这个nn.Sequential()容器中，并按照名称命名为structural_layer_i
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            # 该层的输出维度作为下一层的输入维度
            input_dim = self.structural_layer_config[i]

        # 2: Temporal Attention Layers
        # 结构层最后一层的维度等于时序层的输入维度
        input_dim = self.structural_layer_config[-1]
        # 同样创建一个空的神经网络序列容器
        temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual)
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]

        return structural_attention_layers, temporal_attention_layers

    def get_loss(self, feed_dict):
        node_1, node_2, node_2_negative, graphs = feed_dict.values()
        # run gnn
        # 最终的节点embedding
        final_emb = self.forward(graphs)  # [N, T, F]
        self.graph_loss = 0
        # 遍历每个时间步
        for t in range(self.num_time_steps):
            # 获取当前时间步t的所有节点embedding
            emb_t = final_emb[:, t, :].squeeze()  # [N, F] = [143, 128]
            # 当前时间步所有起始节点的embedding [180, 128]
            source_node_emb = emb_t[node_1[t]]
            # 当前时间步所有上下文终止节点的embedding（正样本） [180, 128]
            tart_node_pos_emb = emb_t[node_2[t]]
            # 当前时间步所有负样本终止节点的embedding [180, 10, 128]
            tart_node_neg_emb = emb_t[node_2_negative[t]]
            pos_score = torch.sum(source_node_emb * tart_node_pos_emb, dim=1)  # 内积并求和 [180]
            neg_score = -torch.sum(source_node_emb[:, None, :] * tart_node_neg_emb, dim=2).flatten()  # [1800]
            pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = self.bceloss(neg_score, torch.ones_like(neg_score))  # 前面得分加了负号，这里标签设为1；或者前面不加负号，这里标签设为0
            graphloss = pos_loss + self.args.neg_weight * neg_loss
            self.graph_loss += graphloss
        return self.graph_loss
