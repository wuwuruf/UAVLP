import numpy as np
import copy
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from utils_DySAT.random_walk import Graph_RandomWalk

import torch

"""Random walk-based pair generation."""


def run_random_walks_n2v(graph, adj, num_walks, walk_len):
    """ In: Graph and list of nodes
        Out: (target, context) pairs from random walk sampling using 
        the sampling strategy of node2vec (deepwalk)
        返回当前图上的随机游走共现对（字典）"""
    # 创建用于随机游走的图（普通图，不保留重复边）
    nx_G = nx.Graph()
    for e in graph.edges():
        # 加边、加节点，此处只保留了有连接的节点（在构图时，将之前的节点加入了）
        nx_G.add_edge(e[0], e[1])
    for edge in graph.edges():
        # 设置边权重（按在多重图中该边的出现次数）
        nx_G[edge[0]][edge[1]]['weight'] = adj[edge[0], edge[1]]

    # node2vec随机游走
    # 实例化一个用于随机游走的类
    G = Graph_RandomWalk(nx_G, False, 1.0, 1.0)
    # 游走概率计算
    G.preprocess_transition_probs()
    # 随机游走
    # walks是列表，长度为当前图的节点数*num_walks，其中的元素是游走序列（列表），长度为walk_len
    walks = G.simulate_walks(num_walks, walk_len)
    # 设置游走序列上的窗口大小，用来获取上下文节点对
    # WINDOW_SIZE的含义是中心节点的左右各延伸的窗口的大小，实际最大的窗口大小为2*WINDOW_SIZE + 1
    WINDOW_SIZE = 10
    pairs = defaultdict(list)
    pairs_cnt = 0
    # 遍历游走序列
    for walk in walks:
        # 遍历游走序列中的节点作为中心节点
        for word_index, word in enumerate(walk):
            # 遍历窗口内元素
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
                # 若当前节点不等于中心节点，则可以作为一对训练语料
                if nb_word != word:
                    # 键为中心节点，值为该中心节点的共现节点列表
                    pairs[word].append(nb_word)
                    # 共现对计数器+1
                    pairs_cnt += 1
    print("# nodes with random walk samples: {}".format(len(pairs)))
    print("# sampled pairs: {}".format(pairs_cnt))
    return pairs


def fixed_unigram_candidate_sampler(true_clasees,
                                    num_true,
                                    num_sampled,
                                    unique,
                                    distortion,
                                    unigrams):
    # TODO: implementate distortion to unigrams
    assert true_clasees.shape[1] == num_true
    samples = []
    # 遍历正样本共现节点，针对每个正样本节点，进行负采样正样本数量的负样本
    for i in range(true_clasees.shape[0]):
        # 深拷贝每个节点的度
        dist = copy.deepcopy(unigrams)
        # 候选节点
        candidate = list(range(len(dist)))
        # 取第i个正样本节点，将tensor类型转为cpu上的list
        taboo = true_clasees[i].cpu().tolist()
        for tabo in sorted(taboo, reverse=True):
            # 删除候选集中的该正样本节点（按值删除）
            candidate.remove(tabo)
            # 删除度列表中该正样本节点对应的度（按索引删除）
            dist.pop(tabo)
        # 每个正样本按概率采样num_sampled个负样本节点
        # replace=False表示无放回的抽取，按照每个候选节点的度抽取，度越大概率越高，抽取num_sampled个
        sample = np.random.choice(candidate, size=num_sampled, replace=unique, p=dist / np.sum(dist))
        samples.append(sample)
    return samples


def to_device(batch, device):
    feed_dict = copy.deepcopy(batch)
    node_1, node_2, node_2_negative, graphs = feed_dict.values()
    # to device
    feed_dict["node_1"] = [x.to(device) for x in node_1]
    feed_dict["node_2"] = [x.to(device) for x in node_2]
    feed_dict["node_2_neg"] = [x.to(device) for x in node_2_negative]
    feed_dict["graphs"] = [g.to(device) for g in graphs]

    return feed_dict
