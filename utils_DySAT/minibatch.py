from typing import DefaultDict
from collections import defaultdict
from torch.functional import Tensor
from torch_geometric.data import Data
from utils_DySAT.utilities import fixed_unigram_candidate_sampler
import torch
import numpy as np
import torch_geometric as tg
import scipy.sparse as sp

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, args, graphs, features, adjs, context_pairs):
        super(MyDataset, self).__init__()
        self.args = args
        # 所有图快照
        self.graphs = graphs
        # 每个图快照的归一化特征
        self.features = [self._preprocess_features(feat) for feat in features]
        # 每个图快照的GCN规范化邻接矩阵
        self.adjs = [self._normalize_graph_gcn(a) for a in adjs]
        self.time_steps = args.time_steps
        self.context_pairs = context_pairs
        # 负采样数量——————————————————————————啥意思？？
        self.max_positive = args.neg_sample_size
        # 最后一个快照的节点（所有节点）
        self.train_nodes = list(self.graphs[self.time_steps - 1].nodes())  # all nodes in the graph.
        # 初始时间步
        # self.min_t = max(self.time_steps - self.args.window - 1, 0) if args.window > 0 else 0
        self.min_t = 0
        # 计算每个图快照中节点的度
        self.degs = self.construct_degs()
        # 获取pyg格式的图列表
        self.pyg_graphs = self._build_pyg_graphs()
        # 创建训练语料
        self.__createitems__()

    def _normalize_graph_gcn(self, adj):
        """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format
           对邻接矩阵进行基于GCN的规范化，以元组的格式返回"""
        adj = sp.coo_matrix(adj, dtype=np.float32)  # 转为稀疏矩阵
        adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)  # 邻接矩阵加上单位阵
        rowsum = np.array(adj_.sum(1), dtype=np.float32)  # 计算加上单位矩阵后的邻接矩阵每行的元素和
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)  # 计算度矩阵的倒数平方根，构建对角矩阵
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()  # 得到GCN规范化后的邻接矩阵
        return adj_normalized

    def _preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation
           对特征矩阵进行行标准化，并转为元组表示"""
        features = np.array(features.todense())  # 将稀疏矩阵转为numpy
        rowsum = np.array(features.sum(1))  # 按行求和
        r_inv = np.power(rowsum, -1).flatten()  # 取行和的倒数
        r_inv[np.isinf(r_inv)] = 0.  # 无穷的倒数赋值为0
        r_mat_inv = sp.diags(r_inv)  # 转成对角阵
        features = r_mat_inv.dot(features)  # 特征归一化
        return features

    def construct_degs(self):
        """ Compute node degrees in each graph snapshot."""
        # different from the original implementation
        # degree is counted using multi graph
        degs = []
        # 遍历每个时间步
        for i in range(self.min_t, self.time_steps):
            G = self.graphs[i]
            deg = []
            # 遍历当前时间步的图的每个节点
            for nodeid in G.nodes():
                deg.append(G.degree(nodeid))
            degs.append(deg)
        # 返回度的列表的列表，degs的元素是列表，是每个图的各节点的度
        return degs

    def _build_pyg_graphs(self):
        pyg_graphs = []
        # 遍历特征和邻接矩阵
        for feat, adj in zip(self.features, self.adjs):
            # 将特征转换为Tensor格式
            x = torch.Tensor(feat)
            # 通过pyg将稀疏矩阵转换为index和weight的格式
            # edge_index, edge_weight都是Tensor；edge_index为两个数组，数组1和数组2的对应节点有连接；edge_weight为对应权重
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
            # 定义pyg的Data
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
            pyg_graphs.append(data)
        # 返回pyg格式的图列表
        return pyg_graphs

    # 所有节点的数量
    def __len__(self):
        return len(self.train_nodes)

    # 获取第index个节点的所有信息
    def __getitem__(self, index):
        node = self.train_nodes[index]
        return self.data_items[node]

    def __createitems__(self):
        self.data_items = {}
        # 遍历最后一张图的节点（所有节点）
        for node in list(self.graphs[self.time_steps - 1].nodes()):
            feed_dict = {}
            node_1_all_time = []  # 中心节点
            node_2_all_time = []  # 共现节点
            # 遍历所有时间步
            for t in range(self.min_t, self.time_steps):
                node_1 = []  # 中心节点
                node_2 = []  # 共现节点
                # 若该时间步的该节点的上下文共现节点数量超过self.max_positive，则随机选择self.max_positive个上下文对作为正样本
                if len(self.context_pairs[t][node]) > self.max_positive:
                    node_1.extend([node] * self.max_positive)
                    node_2.extend(np.random.choice(self.context_pairs[t][node], self.max_positive, replace=False))
                # 否则，将所有上下文对添加为正样本
                else:
                    node_1.extend([node] * len(self.context_pairs[t][node]))
                    node_2.extend(self.context_pairs[t][node])
                assert len(node_1) == len(node_2)
                # 拼上每个时间步的正样本
                node_1_all_time.append(node_1)
                node_2_all_time.append(node_2)

            # 类型转换，把其中每个时间步的列表转为LongTensor类型
            node_1_list = [torch.LongTensor(node) for node in node_1_all_time]
            node_2_list = [torch.LongTensor(node) for node in node_2_all_time]
            # 负样本
            node_2_negative = []
            # 每个时间步进行负采样
            for t in range(len(node_2_list)):
                # 该时刻每个节点的度
                degree = self.degs[t]
                # 正样本节点
                # 取第t个时间步的共现节点数据张量，在列维度上增加一个维度，相当于给每个元素添加一个额外的维度，将原本一维张量扩展为二维张量
                node_positive = node_2_list[t][:, None]
                node_negative = fixed_unigram_candidate_sampler(true_clasees=node_positive,
                                                                num_true=1,
                                                                num_sampled=self.args.neg_sample_size,
                                                                unique=False,
                                                                distortion=0.75,
                                                                unigrams=degree)
                node_2_negative.append(node_negative)
            # 一个正样本对应十个负样本
            node_2_neg_list = [torch.LongTensor(node) for node in node_2_negative]

            feed_dict['node_1'] = node_1_list  # 中心节点（就是该节点）
            feed_dict['node_2'] = node_2_list  # 该中心节点的上下文共现节点（正样本）
            feed_dict['node_2_neg'] = node_2_neg_list  # 负样本
            feed_dict["graphs"] = self.pyg_graphs  # pyg格式的图列表

            # 该节点对应的所有信息
            self.data_items[node] = feed_dict


    @staticmethod
    # 书签知乎收藏解释collate_fn函数
    # 在Dataset的__getitem__把一条一条的数据发出来以后
    # Dataloader会根据你定义的batch_size参数把这些东西组织起来（其实是一个batch_list）
    # 然后再送给collate_fn组织成batch最后的样子
    def collate_fn(samples):
        batch_dict = {}
        for key in ["node_1", "node_2", "node_2_neg"]:
            data_list = []
            # 遍历每一个节点的信息
            for sample in samples:
                data_list.append(sample[key])
            concate = []
            # 遍历每个时间步
            for t in range(len(data_list[0])):
                # 对于所有节点，都选择t这个时间步中的节点信息
                concate.append(torch.cat([data[t] for data in data_list]))
            # key下的所有时间涉及到的节点
            batch_dict[key] = concate
        batch_dict["graphs"] = samples[0]["graphs"]
        return batch_dict  # 每个类别下，所有时间步中涉及到的节点
