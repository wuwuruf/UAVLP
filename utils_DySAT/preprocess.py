import numpy as np
import dill
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from utils_DySAT.utilities import run_random_walks_n2v

np.random.seed(123)


def load_graphs(dataset_str):
    """Load graph snapshots given the name of dataset"""
    with open("data/{}/{}".format(dataset_str, "graph.pkl"), "rb") as f:
        graphs = pkl.load(f)
    print("Loaded {} graphs ".format(len(graphs)))
    adjs = [nx.adjacency_matrix(g) for g in graphs]
    return graphs, adjs


def get_context_pairs(graphs, adjs):
    """ Load/generate context pairs for each snapshot through random walk sampling.
        为每个图快照通过随机游走采样来生成节点对"""
    print("Computing training pairs ...")
    # 上下文节点对列表，列表中的元素是字典，字典的键为中心节点
    context_pairs_train = []
    # 遍历每张图快照，进行node2vec随机游走
    for i in range(len(graphs)):
        context_pairs_train.append(run_random_walks_n2v(graphs[i], adjs[i], num_walks=10, walk_len=40))

    return context_pairs_train


def get_evaluation_data(graphs):
    """ Load train/val/test examples to evaluate link prediction performance
        加载链路预测的train/val/test集"""
    # 获取倒数第二张图的index
    eval_idx = len(graphs) - 2
    # 获取倒数第二张图和最后一张图
    eval_graph = graphs[eval_idx]
    next_graph = graphs[eval_idx + 1]
    print("Generating eval data ....")
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        create_data_splits(eval_graph, next_graph, val_mask_fraction=0.2,
                           test_mask_fraction=0.6)

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def create_data_splits(graph, next_graph, val_mask_fraction=0.2, test_mask_fraction=0.6):
    # 获取下一张图（最后一张图）的边的numpy数组
    edges_next = np.array(list(nx.Graph(next_graph).edges()))
    edges_positive = []  # Constraint to restrict new links to existing nodes.
    # 遍历下一张图的边
    for e in edges_next:
        # 如果当前图拥有该下一张图的边的终端节点，则把该边加入正样本
        if graph.has_node(e[0]) and graph.has_node(e[1]):
            edges_positive.append(e)
    # 转为numpy数组（也没用上啊）
    edges_positive = np.array(edges_positive)  # [E, 2]
    # 负采样，返回负样本列表
    edges_negative = negative_sample(edges_positive, graph.number_of_nodes(), next_graph)

    # from sklearn.model_selection import train_test_split
    # 划分训练集，测试集，验证集
    # 首先划分训练集和测试集（包含验证集）,test_size为测试集（包含验证集）所占总样本数的比例
    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive,
                                                                            edges_negative,
                                                                            test_size=val_mask_fraction + test_mask_fraction)
    # 然后划分测试集为测试集与验证集，test_size为最终测试集所占原测试集的比例
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos,
                                                                                    test_neg,
                                                                                    test_size=test_mask_fraction / (
                                                                                                test_mask_fraction + val_mask_fraction))

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg


def negative_sample(edges_pos, nodes_num, next_graph):
    edges_neg = []
    # 采样与正样本数量相同的负样本边
    while len(edges_neg) < len(edges_pos):
        # 随机采样
        idx_i = np.random.randint(0, nodes_num)
        idx_j = np.random.randint(0, nodes_num)
        # 自连接不采
        if idx_i == idx_j:
            continue
        # 存在的边不采
        if next_graph.has_edge(idx_i, idx_j) or next_graph.has_edge(idx_j, idx_i):
            continue
        # 若负样本不为空，已采样的边不采
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:
                continue
        edges_neg.append([idx_i, idx_j])
    return edges_neg
