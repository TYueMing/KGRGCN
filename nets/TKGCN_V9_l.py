import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, TopKPooling, GCNConv , GATConv, FastRGCNConv
from torch_geometric.nn import global_mean_pool as gap
from torch.nn import Linear
from torch_geometric.utils import softmax
from torch_geometric.nn.pool.connect.filter_edges import filter_adj

'''
GCNConv: 一种经典图卷积操作，基于谱卷积理论，核心：通过邻居矩阵来聚合节点的邻居信息。在聚合过程中，每一个节点只会去聚合周围邻居节点。

RGCNConv: 是GCN扩展，能够处理多个边类型的图，分别学习每种边类型的聚合权重。

GATConv: 一种基于注意力的卷积方法，在聚合邻居节点的时候会自适应的权重加权每一节点信息，

'''

# 一个基于关系图卷积网络（FastRGCNConv）和自注意力机制的操作，用于图神经网络中的节点聚合和降维。
class RGCNSA(nn.Module):
    def __init__(self, in_channels, num_relations, min_score=None, nonlinearity=torch.tanh, **kwargs):
        super(RGCNSA, self).__init__()
        # **kwargs: 允许传递额外参数，用于配置 FastRGCNConv
        self.in_channels = in_channels   # in_channels: 输入特征的维度（通道数）
        self.frgnn = FastRGCNConv(in_channels, 1, num_relations, **kwargs)  # self.gnn: 初始化 FastRGCNConv 对象，这是一个关系图卷积网络层，用于计算节点的注意力分数。输出通道数为 1，因为只需要一个分数值。
        self.min_score = min_score  # min_score: 如果指定，则用于对节点进行最小分数过滤
        self.nonlinearity = nonlinearity    # nonlinearity: 应用于节点得分的非线性激活函数，默认是 torch.tanh
        self.reset_parameters()   # 调用 reset_parameters 方法，重置 FastRGCNConv 的参数

    def reset_parameters(self):
        # 通过调用 self.gnn.reset_parameters() 来重置 FastRGCNConv 层的参数。通常在模型初始化或重置训练时使用。
        self.frgnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        if batch is None:   # batch: 批处理索引，如果 None 则初始化为零张量
            # 如果 batch 没有提供，则创建一个零张量，表示所有节点属于同一批次。
            batch = edge_index.new_zeros(x.size(0))     # [0,0,0, ]   159个0

        attn = x if attn is None else attn   # attn: 如果 attn 为 None，则使用输入特征 x
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn   # 如果 attn 是 1 维张量，则添加一个维度，使其成为 2 维张量
        score = self.frgnn(attn, edge_index, edge_attr).view(-1)  # 通过 FastRGCNConv 计算每个节点的注意力分数（输出形状为 [num_nodes, 1]），然后使用 .view(-1) 展平为 1 维张量

        if self.min_score is None:   # 如果 self.min_score 未设置，则对 score 应用非线性函数（例如 tanh）
            score = self.nonlinearity(score)    ###### 这会将输入的每个数值压缩到区间 [−1,1]，也就是说，输出值会被限制在 -1 到 1 之间
        else:
            score = softmax(score, batch)

        x = x* score.view(-1, 1)   # 根据 perm 选择节点特征 x，并乘以对应的 score

        return x, score

    def __repr__(self):    # 定义对象的字符串表示形式，主要用于打印模型结构
        return '{}({}, {})'.format(
            self.__class__.__name__,
            self.frgnn.__class__.__name__,
            self.in_channels
        )



######################################### Transformer #######################################
class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.2):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        attn_output, _ = self.attention(Q, K, V)    # attn_output.shape = [1, 159, 64]
        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + Q)  # residual connection

        ff_output = self.fc(attn_output)     #  ff_output.shape = [1, 159, 64]
        ff_output = self.dropout(ff_output)
        output = self.layer_norm(ff_output + attn_output)  # residual connection

        return output


######################################### Transformer Decoder #######################################
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.2):
        super(TransformerDecoder, self).__init__()
        self.attention1 = nn.MultiheadAttention(hidden_dim, num_heads, dropout)    # 自注意力
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attention2 = nn.MultiheadAttention(hidden_dim, num_heads, dropout)  # 自注意力
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, Q , q_encoder):
        attn_output, _ = self.attention1(Q, Q, Q)
        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + Q)

        attn_output, _ = self.attention2(attn_output, attn_output, q_encoder)
        attn_output = self.dropout2(attn_output)
        attn_output = self.layer_norm2(attn_output + q_encoder)

        ff_output = self.fc(attn_output)  # ff_output.shape = [1, 159, 64]
        ff_output = self.dropout3(ff_output)
        output = self.layer_norm3(ff_output + attn_output)  # residual connection

        return output


######################################### TKGCN #######################################
######################################### V1:只有结构，没有利用关系的设计 #######################################
class TKGCN_V9(nn.Module):
    def __init__(self,
                 nhid,
                 pooling_rate,
                 dropout_rate,
                 num_classes,
                 num_features,
                 num_relations):
        super(TKGCN_V9, self).__init__()

        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.pooling_ratio = pooling_rate
        self.dropout_ratio = dropout_rate
        self.readout_type = 'mean'
        self.activation = F.leaky_relu

        self.nhid_structre = nhid

        ############################### 结构提取
        self.structure_rgcn1 = RGCNConv(in_channels=num_features, out_channels=self.nhid_structre, num_relations=num_relations)
        self.structure_rgcn2 = RGCNConv(in_channels=self.nhid_structre, out_channels=self.nhid_structre, num_relations=num_relations)
        self.structure_rgcn3 = RGCNConv(in_channels=self.nhid_structre, out_channels=self.nhid_structre, num_relations=num_relations)


        ################################# 主线提取特征
        self.feature_rgcn = RGCNConv(in_channels=num_features, out_channels=nhid, num_relations=num_relations)
        self.feature_rgcn2 = RGCNConv(in_channels=nhid + self.nhid_structre, out_channels=nhid, num_relations=num_relations)
        self.feature_rgcn3 = RGCNConv(in_channels=nhid + self.nhid_structre, out_channels=nhid,
                                      num_relations=num_relations)
        self.feature_rgcn4 = RGCNConv(in_channels=nhid + self.nhid_structre, out_channels=nhid + self.nhid_structre,
                                      num_relations=num_relations)
        self.conv_SA = RGCNSA(self.nhid+self.nhid_structre, num_relations, min_score=None)


        self.transformer = TransformerEncoder(self.nhid + self.nhid_structre, num_heads=1, dropout=self.dropout_ratio)

        self.transformer_decoder = TransformerDecoder(self.nhid + self.nhid_structre, num_heads=1, dropout=self.dropout_ratio)
        # 最终分类层
        self.fc = nn.Linear(self.nhid+self.nhid_structre, self.num_classes)


    # 前向传播
    def forward(self, x, edge_index, edge_attr,batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            # # 获取输入数据
        # x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # # 提取边类型    edge_attr是独热编码后的四维度的，而RGCN要求输入的是1维度
        # edge_type = edge_attr.argmax(dim=1)
        edge_type = edge_attr



        x_main_lane = x
        ############### 空间结构提取 ##############
        x = self.activation(self.structure_rgcn1(x, edge_index, edge_type))
        x1 = x

        x = self.activation(self.structure_rgcn2(x, edge_index, edge_type))
        x2 = x

        x = self.activation(self.structure_rgcn3(x, edge_index, edge_type))
        x3 = x


        # structure_2= torch.cat([x1, x2], dim=-1)
        structure_1 = gap(x1, batch)
        structure_2 = gap(x2, batch)
        structure_3 = gap(x3, batch)


        structure_expanded_1 = structure_1[batch]
        structure_expanded_2 = structure_2[batch]
        structure_expanded_3 = structure_3[batch]

        structure_expanded_1 = structure_expanded_1 + x1
        structure_expanded_2 = structure_expanded_2 + x2
        structure_expanded_3 = structure_expanded_3 + x3


        ############### 主线特征提取 ##############
        x_main_lane = self.activation(self.feature_rgcn(x_main_lane, edge_index, edge_type))
        x_main_lane = torch.cat([x_main_lane,structure_expanded_1],dim=-1)

        x_main_lane = self.activation(self.feature_rgcn2(x_main_lane, edge_index, edge_type))
        x_main_lane = torch.cat([x_main_lane, structure_expanded_2], dim=-1)

        x_main_lane = self.activation(self.feature_rgcn3(x_main_lane, edge_index, edge_type))
        Q_decoder = torch.cat([x_main_lane, structure_expanded_3], dim=-1)
        #
        # Q_decoder, q_scores = self.conv_SA(x_main_lane, edge_index, edge_attr=edge_type, batch=batch)
        Q_decoder = self.activation(self.feature_rgcn4(x_main_lane, edge_index, edge_type))

        # Q = torch.cat([x_main_lane,structure_expanded],dim=-1)

        ############### transformer提取 ##############
        # 为了适配 MultiheadAttention，调整维度为 [N, 1, hidden_dim]
        Q_decoder = Q_decoder.unsqueeze(1).transpose(0, 1)  # [1, N, hidden_dim]   Q_gcn.shape= [1, 159, 64]

        # Q_encoder = Q_encoder.unsqueeze(1).transpose(0, 1)

        # 使用 Transformer 进行处理
        x = self.transformer(Q_decoder, Q_decoder, Q_decoder)
        #
        # x = self.transformer_decoder(Q_decoder,Q_decoder)

        # 获取最终的节点表示
        x = x.squeeze(0)

        # 图池化
        x = gap(x, batch)

        # 分类
        x = self.fc(x)
        # x = F.log_softmax(x, dim=-1)

        return x













