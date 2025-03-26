from .base_strategy import BaseStrategy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli
from tqdm import tqdm
class MHPLSampling(BaseStrategy):

    def query(self, n_select):
        self.net = self.net[0]

        loader_list = self.build_divided_unlabel_loader(self.active_cfg.sub_num)
        total_NAU = []
        total_topk = []
        # uncertainty = []
        for i, unlabel_loader in enumerate(loader_list):
            print('unlabel loader: %d/%d' % (i + 1, len(loader_list)))

            all_probs, all_embs, _, _ = self.predict_probs_and_embed(unlabel_loader)
            all_embs = F.normalize(all_embs, dim=-1)

            # probs, embeddings = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            sim = all_embs.cpu().mm(all_embs.transpose(1, 0))
            K = self.strategy_cfg.S_K
            sim_topk, topk = torch.topk(sim, k=K + 1, dim=1)
            sim_topk, topk = sim_topk[:, 1:], topk[:, 1:]
            total_topk.append(topk)

            # get NP scores
            all_preds = all_probs.argmax(-1)
            Sp = (torch.eye(self.net.n_label)[all_preds[topk]]).sum(1)
            Sp = Sp / Sp.sum(-1, keepdim=True)
            NP = -(torch.log(Sp + 1e-9) * Sp).sum(-1)

            # get NA scores
            NA = sim_topk.sum(-1) / K
            NAU = NP * NA
            total_NAU.append(NAU)
        total_NAU = torch.cat(total_NAU, dim=0)
        total_topk = torch.cat(total_topk, dim=0)



        # unlabel_loader, unlabel_ind = self.build_unlabel_loader()
        #
        # all_probs, all_embs, _, _ = self.predict_probs_and_embed(unlabel_loader)
        # all_embs = F.normalize(all_embs, dim=-1)
        #
        # # probs, embeddings = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        # sim = all_embs.cpu().mm(all_embs.transpose(1, 0))
        # K = self.strategy_cfg.S_K
        # sim_topk, topk = torch.topk(sim, k=K + 1, dim=1)
        # sim_topk, topk = sim_topk[:, 1:], topk[:, 1:]
        #
        # # get NP scores
        # all_preds = all_probs.argmax(-1)
        # Sp = (torch.eye(self.net.n_label)[all_preds[topk]]).sum(1)
        # Sp = Sp / Sp.sum(-1, keepdim=True)
        # NP = -(torch.log(Sp + 1e-9) * Sp).sum(-1)
        #
        # # get NA scores
        # NA = sim_topk.sum(-1) / K
        # NAU = NP * NA
        sort_idxs = total_NAU.argsort(descending=True)

        q_idxs = []
        ax, rem = 0, n_select
        while rem > 0:
            if total_topk[sort_idxs[ax]][0] not in q_idxs:
                q_idxs.append(sort_idxs[ax])
            rem = n_select - len(q_idxs)
            ax += 1

        q_idxs = np.array(q_idxs)

        return self.label_info.unlabel_ind[q_idxs]
        # return unlabel_ind[select_inds]




