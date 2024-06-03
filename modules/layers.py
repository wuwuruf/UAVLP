import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as Init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphNeuralNetwork(Module):
    '''
    Class to define the GNN layer (w/ sparse matrix multiplication)
    '''

    def __init__(self, input_dim, output_dim, dropout_rate):  # 声明（创建模型）的时候用这个
        super(GraphNeuralNetwork, self).__init__()
        # ====================
        self.input_dim = input_dim  # Dimensionality of input features
        self.output_dim = output_dim  # Dimensionality of output features
        self.dropout_rate = dropout_rate  # Dropout rate
        # ====================
        # Initialize model parameters
        self.agg_wei = Init.xavier_uniform_(
            Parameter(torch.FloatTensor(input_dim, output_dim)))  # Aggregation weight matrix  参数一般用FloatTensor类型
        # ==========
        self.param = nn.ParameterList()
        self.param.append(self.agg_wei)
        # ==========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, feat, sup):  # 前向传播（创建实例）的时候用这个
        '''
        Rewrite the forward function
        :param feat: feature input of the GCN layer
        :param sup: GCN support (normalized adjacency matrix) sup矩阵是A波浪
        :return: aggregated feature output of the GCN layer
        '''
        # ====================
        # Feature aggregation from immediate neighbors
        feat_agg = torch.spmm(sup, feat)  # Aggregated feature  输入的sup是稀疏矩阵，因此用spmm
        agg_output = torch.relu(torch.mm(feat_agg, self.param[0]))
        agg_output = F.normalize(agg_output, dim=1, p=2)  # l2-normalization  使用L2范数对聚合后的特征进行归一化，特征变为单位向量
        agg_output = self.dropout_layer(agg_output)

        return agg_output


class GraphNeuralNetworkDense(Module):
    '''
    Class to define the GNN layer (w/ dense matrix multiplication)
    '''

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(GraphNeuralNetworkDense, self).__init__()
        # ====================
        self.input_dim = input_dim  # Dimensionality of input features
        self.output_dim = output_dim  # Dimensionality of output features
        self.dropout_rate = dropout_rate  # Dropout rate
        # ====================
        # Initialize model parameters
        self.agg_wei = Init.xavier_uniform_(
            Parameter(torch.FloatTensor(input_dim, output_dim)))  # Aggregation weight matrix
        # ==========
        self.param = nn.ParameterList()
        self.param.append(self.agg_wei)
        # =========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, feat, sup):
        '''
        Rewrite the forward function
        :param feat: feature input of the GCN layer
        :param sup: GCN support (normalized adjacency matrix)
        :return: aggregated feature output of the GCN layer
        '''
        # ====================
        # Feature aggregation from immediate neighbors
        feat_agg = torch.mm(sup, feat)  # Aggregated feature
        agg_output = torch.relu(torch.mm(feat_agg, self.param[0]))
        agg_output = F.normalize(agg_output, dim=1, p=2)  # l2-normalization
        agg_output = self.dropout_layer(agg_output)

        return agg_output


class AttNodeAlign(Module):
    '''
    Class to define attentive node aligning unit
    '''

    def __init__(self, feat_dim, hid_dim, dropout_rate):
        super(AttNodeAlign, self).__init__()
        # ====================
        self.dropout_rate = dropout_rate
        # ====================
        self.feat_dim = feat_dim  # Dimensionality of feature input
        self.hid_dim = hid_dim  # Dimensionality of the hidden space
        # ====================
        self.from_map = nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)  # FC feature mapping
        self.to_map = nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)  # FC feature mapping
        # =========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, align, feat_from, feat_to, emb, lambd):
        '''
        Rewrite the forward function
        :param align: align matrix
        :param feat_from: (reduced) GNN feature of timeslice t
        :param feat_to: (reduced) GNN feature of timeslice (t+1)
        :param emb: hidden embedding
        :param lambd: factor of the attention module
        :return: aligned features
        '''
        # ====================
        feat_from_ = torch.tanh(self.from_map(feat_from))
        feat_to_ = torch.tanh(self.to_map(feat_to))
        att_align = torch.mm(feat_from_, feat_to_.t())
        hyd_align = align + lambd * att_align
        feat_align = torch.mm(hyd_align.t(), emb)

        return feat_align


class BiGNNAlign(Module):
    def __init__(self, feat_dim, hid_dim, dropout_rate):
        super(BiGNNAlign, self).__init__()
        # ====================
        self.dropout_rate = dropout_rate
        # ====================
        self.feat_dim = feat_dim  # Dimensionality of feature input
        self.hid_dim = hid_dim  # Dimensionality of the hidden space
        # ====================
        self.from_map = nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)  # FC feature mapping
        self.to_map = nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)  # FC feature mapping
        # =========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, align, feat_from, feat_to, emb, fact):
        # ====================
        feat_from_ = torch.tanh(self.from_map(feat_from))  # relu, sigmoid
        feat_to_ = torch.tanh(self.to_map(feat_to))  # relu, sigmoid
        att_align = torch.mm(feat_from_, feat_to_.t())
        hyd_align = align + fact * att_align
        feat_align = torch.mm(hyd_align.t(), emb)

        return feat_align


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


# =======================================================================================
class TemporalAttentionLayer(nn.Module):
    def __init__(self,
                 input_dim,  # 输入维度=输出维度
                 n_heads,
                 num_time_steps,
                 dropout_rate):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        # self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(dropout_rate)
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
        # if self.residual:
        #     outputs = outputs + temporal_inputs
        # return outputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        # return outputs + inputs
        return outputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)