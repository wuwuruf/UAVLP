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

    def __init__(self, micro_dims, agg_feat_dim, RNN_dims, decoder_dims,
                 n_heads,
                 dropout_rate):
        super(MultiAggLP, self).__init__()

        self.micro_dims = micro_dims  # 学习微观表示的GAT层的维度
        # self.meso_dims = meso_dims  # 介观池化层的维度
        # self.macro_dims = macro_dims  # 宏观池化层的维度
        self.agg_feat_dim = agg_feat_dim  # 聚合得到的特征维度
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
                WeightedGAT(input_dim=self.micro_dims[l], output_dim=self.micro_dims[l + 1], n_heads=self.n_heads,
                            drop_rate=self.dropout_rate))
        # ==================
        # 学习宏观池化表示
        self.macro_pooling_layers = WeiPool_noW()
        # ==================
        # 学习介观池化表示
        self.meso_pooling_layers = WeiPool_noW()
        # ==================
        # 获取注意力融合表示
        self.agg_layers = AttMultiAgg_guide_att(self.micro_dims[-1], self.dropout_rate)
        # ==================
        # 学习时序信息  先用两层LSTM试试！！！

        self.num_RNN_layers = len(self.RNN_dims) - 1
        self.RNN_layers = nn.ModuleList()
        for l in range(self.num_RNN_layers):
            self.RNN_layers.append(
                ILSTM(input_dim=self.RNN_dims[l], output_dim=self.RNN_dims[l + 1], dropout_rate=self.dropout_rate))
        # ==================
        # 解码器
        self.decoder = FCNN(self.decoder_dims[0], self.decoder_dims[1], self.decoder_dims[2], self.dropout_rate)

    def forward(self, edge_index_list, edge_weight_list, feat_list, D_com_list_list, partition_dict_list,
                D_list,
                pred_flag=True):
        """
        :param edge_weight_list:
        :param edge_index_list:
        :param feat_list:就是 torch.FloatTensor
        :param D_com_list_list:
        :param partition_dict_list: 社团划分结果，key为节点编号，value为社团编号
        :param pred_flag:
        :param D_list:
        :return:
        """
        win_size = len(feat_list)
        num_nodes = feat_list[0].shape[0]
        # =======================
        # 学习微观表示矩阵的列表
        input_micro_feat_list = feat_list
        output_micro_feat_list = None
        for l in range(self.num_micro_GAT_layers):
            micro_layer = self.micro_GAT_layers[l]
            output_micro_feat_list = []
            for t in range(win_size):
                # if l == 0 and t == 9:
                #     print(1)
                output_micro_feat = micro_layer(edge_index_list[t], edge_weight_list[t], input_micro_feat_list[t])
                output_micro_feat_list.append(output_micro_feat)
            input_micro_feat_list = output_micro_feat_list
        # =======================
        # 介观池化获取介观表示矩阵的列表
        output_meso_feat_list = []
        for t in range(win_size):
            partition_dict = partition_dict_list[t]
            output_micro_feat = output_micro_feat_list[t]
            D_com_list = D_com_list_list[t]
            output_meso_feat = torch.empty(num_nodes, self.micro_dims[-1]).to(device)  # 介观特征矩阵
            for com_idx in range(len(D_com_list)):
                cur_com_nodes_list = [key for key, value in partition_dict.items() if
                                      value == com_idx]  # 找出属于当前社团的节点编号列表
                D_com = D_com_list[com_idx]
                # if t == 9 and com_idx == 8:
                #     print(2)  # 第四次的t == 9时
                output_meso_com_feat = self.meso_pooling_layers(output_micro_feat[cur_com_nodes_list],
                                                                D_com)  # 该社团内进行池化得到的特征
                output_meso_feat[cur_com_nodes_list] = output_meso_com_feat  # 将介观特征矩阵的对应当前社团的行直接赋值为当前社团池化特征
            output_meso_feat_list.append(output_meso_feat)
        # =======================
        # 宏观池化获取宏观表示矩阵的列表
        output_macro_feat_list = []
        for t in range(win_size):
            output_macro_feat = self.macro_pooling_layers(output_micro_feat_list[t], D_list[t])
            output_macro_feat = output_macro_feat.expand(num_nodes, -1)  # 扩充列数使其形状与微观表示矩阵相同
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
        # input_RNN_list = output_micro_feat_list
        output_RNN_list = None
        for l in range(self.num_RNN_layers):
            RNN_layer = self.RNN_layers[l]
            output_RNN_list = []
            pre_state = torch.zeros(num_nodes, self.RNN_dims[l + 1]).to(device)
            pre_cell = torch.zeros(num_nodes, self.RNN_dims[l + 1]).to(device)
            for t in range(win_size):
                output_state, output_cell = RNN_layer(pre_state, pre_cell, input_RNN_list[t])
                pre_state = output_state
                pre_cell = output_cell
                output_RNN_list.append(output_state)
            input_RNN_list = output_RNN_list
        # ======================
        # 解码器
        if pred_flag:  # 预测模式，仅预测窗口外下一时刻的快照
            input_feat = F.normalize(output_RNN_list[-1], dim=0, p=2)  # 对嵌入矩阵列向量进行标准化
            # input_feat = output_RNN_list[-1]
            pred_adj = self.decoder(input_feat)
            return [pred_adj]
        else:
            pred_adj_list = []
            for t in range(win_size):
                input_feat = F.normalize(output_RNN_list[t], dim=0, p=2)  # 对嵌入矩阵列向量进行标准化
                # input_feat = output_RNN_list[t]
                pred_adj = self.decoder(input_feat)
                pred_adj_list.append(pred_adj)
            return pred_adj_list
        # ============
