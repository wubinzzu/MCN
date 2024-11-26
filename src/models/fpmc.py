import torch
from torch import nn
from torch.nn.init import xavier_normal_

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss


class FPMC(GeneralRecommender):
    r"""The FPMC model is mainly used in the recommendation system to predict the possibility of
    unknown items arousing user interest, and to discharge the item recommendation list.

    Note:

        In order that the generation method we used is common to other sequential models,
        We set the size of the basket mentioned in the paper equal to 1.
        For comparison with other models, the loss function used is BPR.

    """

    def __init__(self, config, dataset):
        super(FPMC, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        # user embedding matrix
        self.UI_emb = nn.Embedding(self.n_users, self.embedding_size)
        # label embedding matrix
        self.IU_emb = nn.Embedding(self.n_items, self.embedding_size)
        # last click item embedding matrix
        self.LI_emb = nn.Embedding(self.n_items, self.embedding_size)
        # label embedding matrix
        self.IL_emb = nn.Embedding(self.n_items, self.embedding_size)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, user, item_seq, next_item):
        item_last_click = item_seq[:, -1].unsqueeze(1)
        LI_emb = torch.cat((self.LI_emb.weight, torch.zeros(1, self.embedding_size).to(self.device)), dim=0)
        item_seq_emb = LI_emb[item_last_click]  # [b,1,emb]

        user_emb = self.UI_emb(user)
        user_emb = torch.unsqueeze(user_emb, dim=1)  # [b,1,emb]

        iu_emb = self.IU_emb(next_item)
        iu_emb = torch.unsqueeze(iu_emb, dim=1)  # [b,n,emb] in here n = 1

        il_emb = self.IL_emb(next_item)
        il_emb = torch.unsqueeze(il_emb, dim=1)  # [b,n,emb] in here n = 1

        # This is the core part of the FPMC model,can be expressed by a combination of a MF and a FMC model
        #  MF
        mf = torch.matmul(user_emb, iu_emb.permute(0, 2, 1))
        mf = torch.squeeze(mf, dim=1)  # [B,1]
        #  FMC
        fmc = torch.matmul(il_emb, item_seq_emb.permute(0, 2, 1))
        fmc = torch.squeeze(fmc, dim=1)  # [B,1]

        score = mf + fmc
        score = torch.squeeze(score)
        return score

    def calculate_loss(self, interaction):
        user = interaction[0][0]  # 2048
        pos_items = interaction[0][1]  # 2048
        neg_items = interaction[0][2]  # 2048  * neg_item_seqs
        item_seq = interaction[1]  # 2048

        pos_score = self.forward(user, item_seq,  pos_items)
        neg_score = self.forward(user, item_seq,  neg_items)
        loss = self.loss_fct(pos_score, neg_score)
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        item_seq = interaction[2]

        user_emb = self.UI_emb(user)
        all_iu_emb = self.IU_emb.weight
        mf = torch.matmul(user_emb, all_iu_emb.transpose(0, 1))
        all_il_emb = self.IL_emb.weight

        item_last_click = item_seq[:, -1].unsqueeze(1)
        item_seq_emb = self.LI_emb(item_last_click)  # [b,1,emb]
        fmc = torch.matmul(item_seq_emb, all_il_emb.transpose(0, 1))
        fmc = torch.squeeze(fmc, dim=1)
        score = mf + fmc
        return score
