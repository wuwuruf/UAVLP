# _*_ coding : utf-8 _*_
# @Time : 2024/6/5 12:50
# @Author : wfr
# @file : model
# @Project : IDEA
"""
注意是用窗口内的图预测窗口外的图，窗口内表示历史图快照
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

from .layers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiAggLP(Module):
    """
    1. 由GAT获取微观表示
    2. 由社团内的池化获取介观表示
    3. 由整图池化获取宏观表示
    4. 对多尺度表示进行注意力融合
    5. 将融合多尺度表示输入GRU学习时序信息
    5. 使用GRU最后的隐藏状态预测邻接矩阵（值在0到1之间）
    """

    def __init__(self, micro_dims, input_feat_dim, pool_hidden_dim, pooling_ratio, agg_feat_dim, RNN_dims, decoder_dims,
                 n_heads,
                 dropout_rate):
        super(MultiAggLP, self).__init__()

        self.micro_dims = micro_dims  # 学习微观表示的GAT层的维度
        # self.meso_dims = meso_dims  # 介观池化层的维度
        # self.macro_dims = macro_dims  # 宏观池化层的维度
        self.input_feat_dim = input_feat_dim  # 输入特征的维度
        self.pooling_hidden_dim = pool_hidden_dim  # 池化层的隐藏层维度，池化层的输出特征维度等于pool_hidden_dim*2
        self.pooling_ratio = pooling_ratio  # 池化率
        self.agg_feat_dim = agg_feat_dim  # 输出的特征维度
        self.RNN_dims = RNN_dims  # GRU的维度
        self.decoder_dims = decoder_dims  # 解码器的维度
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        # ===================
        # 学习微观表示
        self.num_micro_GAT_layers = len(self.micro_dims) - 1
        self.micro_GAT_layers = nn.ModuleList()
        for l in range(self.num_micro_GAT_layers):
            self.micro_GAT_layers.append(
                WeightedGAT(input_dim=micro_dims[l], output_dim=micro_dims[l + 1], n_heads=self.n_heads,
                            drop_rate=self.dropout_rate))
        # ==================
        # 学习宏观池化表示
        self.macro_pooling_layers = HierarchicalPool(self.input_feat_dim, self.pooling_hidden_dim, self.pooling_ratio,
                                                     self.n_heads,
                                                     dropout_ratio=self.dropout_rate)
        # ==================
        # 学习介观池化表示
        self.meso_pooling_layers = HierarchicalPool(self.input_feat_dim, self.pooling_hidden_dim, self.pooling_ratio,
                                                    self.n_heads,
                                                    dropout_ratio=self.dropout_rate)
        # ==================
        # 获取注意力融合表示
        self.agg_layers = AttMultiAgg(self.pooling_hidden_dim * 2, self.agg_feat_dim)
        # ==================
        # 学习时序信息  先用两层GRU试试！！！
        self.num_RNN_layers = len(self.RNN_dims) - 1
        self.RNN_layers = nn.ModuleList()
        for l in range(self.num_RNN_layers):
            self.RNN_layers.append(
                IGRU(input_dim=self.RNN_dims[l], output_dim=self.RNN_dims[l + 1], dropout_rate=self.dropout_rate))
        # ==================
        # 解码器
        self.decoder = FCNN(self.decoder_dims[0], self.decoder_dims[1], self.decoder_dims[2])

    def forward(self, edge_index_list, edge_weight_list, feat_list, pred_flag=True):
        """
        :param edge_weight_list:
        :param edge_index_list:
        :param feat_list:就是 torch.FloatTensor
        :return:
        """
        win_size = len(feat_list)
        # =======================
        # 学习微观表示矩阵的列表
        input_micro_feat_list = feat_list
        output_micro_feat_list = None
        for l in range(self.num_micro_GAT_layers):
            micro_layer = self.micro_GAT_layers[l]
            output_micro_feat_list = []
            for t in range(win_size):
                output_micro_feat = micro_layer(edge_index_list[t], edge_weight_list[t], input_micro_feat_list[t])
                output_micro_feat_list.append(output_micro_feat)
            input_micro_feat_list = output_micro_feat_list
        # =======================
        # 介观池化获取介观表示矩阵的列表
        output_meso_feat_list = []
        # =======================
        # 宏观池化获取宏观表示矩阵的列表
        output_macro_feat_list = []
        for t in range(win_size):
            output_macro_feat = self.macro_pooling_layers(edge_index_list[t], edge_weight_list[t],
                                                          output_micro_feat_list[t])
            output_macro_feat = output_macro_feat.expand(feat_list[t].shape[0], -1)  # 扩充列数使其形状与微观表示矩阵相同
            output_macro_feat_list.append(output_macro_feat)
        # =======================
        # 对多尺度表示进行融合，获取多尺度表示矩阵的列表
        output_agg_feat_list = []
        for t in range(win_size):
            output_agg_feat = self.agg_layers(output_micro_feat_list[t], output_meso_feat_list[t],
                                              output_macro_feat_list[t])
            output_agg_feat_list.append(output_agg_feat)
        # ======================
        # 学习多尺度表示中的时序特征
        input_RNN_list = output_agg_feat_list
        output_RNN_list = None
        for l in range(self.num_RNN_layers):
            RNN_layer = self.RNN_layers[l]
            output_RNN_list = []
            pre_state = torch.zeros(feat_list.shape[0], self.RNN_dims[l]).to(device)
            for t in range(win_size):
                output_state = RNN_layer(pre_state, input_RNN_list[t])
                pre_state = output_state
                output_RNN_list.append(output_state)
            input_RNN_list = output_RNN_list
        # ======================
        # 解码器
        if pred_flag:  # 预测模式，仅预测窗口外下一时刻的快照
            input_feat = F.normalize(output_RNN_list[-1], dim=0, p=2)  # 对嵌入矩阵行向量进行标准化
            pred_adj = self.decoder(input_feat)
            return [pred_adj]
        else:
            pred_adj_list = []
            for t in range(win_size):
                input_feat = F.normalize(output_RNN_list[t], dim=0, p=2)  # 对嵌入矩阵行向量进行标准化
                pred_adj = self.decoder(input_feat)
                pred_adj_list.append(pred_adj)
            return pred_adj_list
