import numpy as np
import os

import scipy.sparse as sp
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_normal_initialization


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.gelu = torch.nn.GELU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.gelu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class Crossmodal_Transformer(nn.Module):
    def __init__(self, n_layers,
                 hidden_size,
                 layer_norm_eps,
                 n_heads,
                 attn_dropout_prob,
                 hidden_dropout_prob,  # feedforward
                 nc_layers,
                 nc_heads
                 ):
        super(Crossmodal_Transformer, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.nc_layers = nc_layers
        self.nc_heads = nc_heads
        self.hidden_size = hidden_size
        self.layer_norm_eps = layer_norm_eps
        self.attn_dropout_prob = attn_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob

        self.attention_layernorms1 = torch.nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.cross_Q_attention_layernorms = torch.nn.ModuleList()
        self.cross_attention_layers = torch.nn.ModuleList()
        self.cross_forward_layernorms = torch.nn.ModuleList()
        self.cross_forward_layers = torch.nn.ModuleList()

        self.self_attention_layernorms = torch.nn.ModuleList()
        self.self_Q_attention_layernorms = torch.nn.ModuleList()
        self.self_attention_layers = torch.nn.ModuleList()
        self.self_forward_layernorms = torch.nn.ModuleList()
        self.self_forward_layers = torch.nn.ModuleList()

        for _ in range(self.nc_layers):
            Q_new_attn_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            self.cross_Q_attention_layernorms.append(Q_new_attn_layernorm)

            cross_new_attn_layer = torch.nn.MultiheadAttention(self.hidden_size,
                                                               self.nc_heads,
                                                               self.attn_dropout_prob)
            self.cross_attention_layers.append(cross_new_attn_layer)

            cross_new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            self.cross_forward_layernorms.append(cross_new_fwd_layernorm)

            cross_new_fwd_layer = PointWiseFeedForward(self.hidden_size, self.hidden_dropout_prob)
            self.cross_forward_layers.append(cross_new_fwd_layer)

        for _ in range(self.n_layers):
            Q_new_attn_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            self.self_Q_attention_layernorms.append(Q_new_attn_layernorm)

            self_new_attn_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            self.self_attention_layernorms.append(self_new_attn_layernorm)

            self_new_attn_layer = torch.nn.MultiheadAttention(self.hidden_size,
                                                              self.n_heads,
                                                              self.attn_dropout_prob)
            self.self_attention_layers.append(self_new_attn_layer)

            self_new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            self.self_forward_layernorms.append(self_new_fwd_layernorm)

            self_new_fwd_layer = PointWiseFeedForward(self.hidden_size, self.hidden_dropout_prob)
            self.self_forward_layers.append(self_new_fwd_layer)

    def forward(self, Q, input_emb, timeline_mask, attention_mask):
        # 送入模型，进行预测
        input_emb = self.attention_layernorms1(input_emb)
        input_emb = torch.transpose(input_emb, 0, 1)
        for i in range(len(self.cross_attention_layers)):
            Q = torch.transpose(Q, 0, 1)
            Q = self.cross_Q_attention_layernorms[i](Q)
            mha_outputs, _ = self.cross_attention_layers[i](Q, input_emb, input_emb, attn_mask=attention_mask)
            Q = Q + mha_outputs
            Q = torch.transpose(Q, 0, 1)
            Q = self.cross_forward_layernorms[i](Q)
            Q = self.cross_forward_layers[i](Q)
            Q *= ~timeline_mask.unsqueeze(-1)

        input_emb = Q
        # self attention
        for i in range(len(self.self_attention_layers)):
            Q = torch.transpose(Q, 0, 1)
            Q = self.self_Q_attention_layernorms[i](Q)
            input_emb = self.self_attention_layernorms[i](input_emb)
            input_emb = torch.transpose(input_emb, 0, 1)
            mha_outputs, _ = self.self_attention_layers[i](Q, input_emb, input_emb, attn_mask=attention_mask)
            input_emb = Q + mha_outputs
            input_emb = torch.transpose(input_emb, 0, 1)
            input_emb = self.self_forward_layernorms[i](input_emb)
            input_emb = self.self_forward_layers[i](input_emb)
            input_emb *= ~timeline_mask.unsqueeze(-1)
            Q = input_emb

        return input_emb


class MCN(GeneralRecommender):

    def __init__(self, config, dataset):
        super(MCN, self).__init__(config, dataset)

        # load gcn parameters info
        self.gcn_n_layers = config["gcn_n_layers"]
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)  # generate intermediate data

        # load sequential parameters info
        self.c_n_layers = config["c_n_layers"]
        self.c_n_heads = config["c_n_heads"]
        self.s_n_layers = config["s_n_layers"]
        self.s_n_heads = config["s_n_heads"]
        self.mm_n_layers = config["mm_n_layers"]
        self.mm_n_heads = config["mm_n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.initializer_range = config["initializer_range"]

        self.loss_type = config["loss_type"]
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton
        self.cl_weight = config['cl_weight']
        self.ot_weight = config['ot_weight']

        # initialize embedding
        self.v_feat_size = self.v_feat.shape[1]  # 4096
        self.t_feat_size = self.t_feat.shape[1]  # 384
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']  # 64
        self.user_id_embedding = nn.Embedding(self.n_users, self.hidden_size)  # 19445 * 64
        self.padding = self.n_items
        self.item_id_embedding = nn.Embedding(self.n_items, self.hidden_size)  # 7050  * 64
        # self.target_item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=self.padding)  # 7050  * 64
        self.id_position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)  # 50 * 64
        self.v_position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)  # 50  * 64
        self.t_position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)  # 50  * 64

        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        # nn.init.xavier_uniform_(self.target_item_embedding.weight)
        nn.init.xavier_uniform_(self.id_position_embedding.weight)
        nn.init.xavier_uniform_(self.v_position_embedding.weight)
        nn.init.xavier_uniform_(self.t_position_embedding.weight)
        # define layers and loss
        self.id_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.id_trm_Encoder1 = Crossmodal_Transformer(self.s_n_layers,
                                                      self.hidden_size,
                                                      self.layer_norm_eps,
                                                      self.s_n_heads,
                                                      self.attn_dropout_prob,
                                                      self.hidden_dropout_prob,
                                                      self.c_n_layers,
                                                      self.c_n_heads
                                                      )
        self.id_trm_Encoder2 = Crossmodal_Transformer(self.mm_n_layers,
                                                      self.hidden_size,
                                                      self.layer_norm_eps,
                                                      self.mm_n_heads,
                                                      self.attn_dropout_prob,
                                                      self.hidden_dropout_prob,
                                                      self.c_n_layers,
                                                      self.c_n_heads
                                                      )
        self.id_last_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-8)

        self.v_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.v_Linear = nn.Linear(self.v_feat_size, self.hidden_size, bias=False)
        self.v_trm_Encoder1 = Crossmodal_Transformer(self.s_n_layers,
                                                     self.hidden_size,
                                                     self.layer_norm_eps,
                                                     self.s_n_heads,
                                                     self.attn_dropout_prob,
                                                     self.hidden_dropout_prob,
                                                     self.c_n_layers,
                                                     self.c_n_heads
                                                     )
        self.v_trm_Encoder2 = Crossmodal_Transformer(self.s_n_layers,
                                                     self.hidden_size,
                                                     self.layer_norm_eps,
                                                     self.s_n_heads,
                                                     self.attn_dropout_prob,
                                                     self.hidden_dropout_prob,
                                                     self.c_n_layers,
                                                     self.c_n_heads
                                                     )
        self.v_last_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-8)

        self.t_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.t_Linear = nn.Linear(self.t_feat_size, self.hidden_size, bias=False)
        self.t_trm_Encoder1 = Crossmodal_Transformer(self.s_n_layers,
                                                     self.hidden_size,
                                                     self.layer_norm_eps,
                                                     self.s_n_heads,
                                                     self.attn_dropout_prob,
                                                     self.hidden_dropout_prob,
                                                     self.c_n_layers,
                                                     self.c_n_heads
                                                     )
        self.t_trm_Encoder2 = Crossmodal_Transformer(self.s_n_layers,
                                                     self.hidden_size,
                                                     self.layer_norm_eps,
                                                     self.s_n_heads,
                                                     self.attn_dropout_prob,
                                                     self.hidden_dropout_prob,
                                                     self.c_n_layers,
                                                     self.c_n_heads
                                                     )
        self.t_last_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-8)

        self.idu_fuse_weights = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, self.hidden_size)))
        self.idi_fuse_weights = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, self.hidden_size)))
        self.v_fuse_weights = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, self.hidden_size)))
        self.t_fuse_weights = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, self.hidden_size)))

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "BCE":
            pass
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def gcn(self, all_embeddings):
        h = self.item_id_embedding.weight
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.gcn_n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def forward(self, user, item_seq):
        user_id_embeddings = self.user_id_embedding.weight  # 19445 * 64
        item_id_embeddings = self.item_id_embedding.weight  # 7050  * 64

        # light gcn
        user_all_id_embeddings, item_all_id_embeddings = self.gcn(
            torch.cat([user_id_embeddings, item_id_embeddings], dim=0))
        # id
        item_all_id_embeddings = torch.cat((item_all_id_embeddings, torch.zeros(1, self.hidden_size).to(self.device)),
                                           dim=0)
        user_id_emb = user_all_id_embeddings[user]  # 2048  * 64
        item_id_emb = item_all_id_embeddings[item_seq]  # 2048 * 50  * 64
        # v
        v_feat = torch.cat((self.v_feat, torch.zeros(1, self.v_feat_size).to(self.device)), dim=0)
        input_v_feat = self.v_Linear(v_feat[item_seq])  # 2048 * 50  * 64
        # t
        t_feat = torch.cat((self.t_feat, torch.zeros(1, self.t_feat_size).to(self.device)), dim=0)
        input_t_feat = self.t_Linear(t_feat[item_seq])  # 2048 * 50  * 64

        # position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)  # 2048*50
        # mask
        # 如果序列太短，前面等于padding，那么对应的序列embedding也要等于0
        timeline_mask = torch.as_tensor(item_seq == self.padding, dtype=torch.bool, device=self.device)
        # 这个是为了实现论文中提到的前面物品在预测时不能用到后面物品的信息，用mask来实现
        tl = item_id_emb.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))
        # id
        item_id_emb *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
        id_position_embedding = self.id_position_embedding(position_ids)
        input_id_emb = item_id_emb + id_position_embedding
        input_id_emb = self.id_dropout(input_id_emb)

        # v
        input_v_feat *= ~timeline_mask.unsqueeze(-1)
        v_position_embedding = self.v_position_embedding(position_ids)
        input_v_feat = input_v_feat + v_position_embedding
        input_v_feat = self.v_dropout(input_v_feat)

        # t
        input_t_feat *= ~timeline_mask.unsqueeze(-1)
        t_position_embedding = self.t_position_embedding(position_ids)
        input_t_feat = input_t_feat + t_position_embedding
        input_t_feat = self.t_dropout(input_t_feat)

        id_output1 = self.id_trm_Encoder1(input_id_emb, input_v_feat, timeline_mask, attention_mask)
        id_output2 = self.id_trm_Encoder2(input_id_emb, input_t_feat, timeline_mask, attention_mask)
        v_output1 = self.v_trm_Encoder1(input_v_feat, input_id_emb, timeline_mask, attention_mask)
        v_output2 = self.v_trm_Encoder2(input_v_feat, input_t_feat, timeline_mask, attention_mask)
        t_output1 = self.t_trm_Encoder1(input_t_feat, input_id_emb, timeline_mask, attention_mask)
        t_output2 = self.t_trm_Encoder2(input_t_feat, input_v_feat, timeline_mask, attention_mask)

        id_output = (id_output1 + id_output2)[:, -1, :]  # 2048  * 64
        v_output = (v_output1 + v_output2)[:, -1, :]  # 2048  * 64
        t_output = (t_output1 + t_output2)[:, -1, :]  # 2048  * 64

        user_id_temp = torch.mm(self.idu_fuse_weights, user_id_emb.transpose(0, 1)).transpose(0, 1)
        id_temp = torch.mm(self.idi_fuse_weights, id_output.transpose(0, 1)).transpose(0, 1)
        v_temp = torch.mm(self.v_fuse_weights, v_output.transpose(0, 1)).transpose(0, 1)
        t_temp = torch.mm(self.t_fuse_weights, t_output.transpose(0, 1)).transpose(0, 1)
        temp = torch.stack([id_temp + user_id_temp, v_temp + user_id_temp, t_temp + user_id_temp], dim=1)
        temp = torch.exp(temp).squeeze(-1)
        tt_out = temp.sum(dim=-1).unsqueeze(-1).repeat(1, 3)
        attention = (temp / tt_out).unsqueeze(1)
        output = torch.stack([id_output, v_output, t_output], dim=1)
        output = torch.bmm(attention, output).squeeze(1) + user_id_emb

        # output = (id_output + v_output + t_output + user_id_emb)/4

        return output  # 2048  * 64

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0][0]: user list; [0][1]: positive items; [0][2]: negative items
            [1]:item_seq
        :return:
        """
        user = interaction[0][0]  # 2048
        pos_items = interaction[0][1]  # 2048
        neg_items = interaction[0][2]  # 2048  * neg_item_seqs
        item_seq = interaction[1]  # 2048
        output = self.forward(user, item_seq)
        item_embedding = self.item_id_embedding.weight

        if self.loss_type == "BPR":
            pos_items_emb = item_embedding[pos_items]
            neg_items_emb = item_embedding[neg_items]
            pos_score = torch.mul(output, pos_items_emb).sum(dim=1)
            neg_score = torch.mul(output, neg_items_emb).sum(dim=1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:
            logits = torch.matmul(output, item_embedding.transpose(0, 1))
            return self.loss_fct(logits, pos_items)

    def full_sort_predict(self, interaction):
        user = interaction[0]
        item_seq = interaction[2]
        seq_output = self.forward(user, item_seq)
        test_items_emb = self.item_id_embedding.weight
        score = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return score
