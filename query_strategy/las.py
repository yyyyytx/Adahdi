from .base_strategy import BaseStrategy
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import copy


class LASSampling(BaseStrategy):
    '''
    (ICCV2023)Local Context-Aware Active Domain Adaptation
    '''

    def query(self, n):
        self.net = self.net[0]
        self.query_count += 1
        # label_loader, label_ind = self.build_labeled_loader()
        # unlabel_loader, unlabel_ind = self.build_unlabel_loader()

        loader_list = self.build_divided_unlabel_loader(self.active_cfg.sub_num)
        total_dists = []
        total_m_idxs = []
        # uncertainty = []
        acc_count = 0
        for i, unlabel_loader in enumerate(loader_list):
            print('unlabel loader: %d/%d' % (i + 1, len(loader_list)))
            all_probs, all_embs, _, _ = self.predict_probs_and_embed(unlabel_loader)
            # get Q_score
            sim = all_embs.cpu().mm(all_embs.transpose(1, 0))
            K = self.strategy_cfg.S_K
            sim_topk, topk = torch.topk(sim, k=K + 1, dim=1)
            sim_topk, topk = sim_topk[:, 1:], topk[:, 1:]
            wgt_topk = (sim_topk / sim_topk.sum(dim=1, keepdim=True))

            Q_score = -((all_probs[topk] * all_probs.unsqueeze(1)).sum(-1) * wgt_topk).sum(-1)
            # propagate Q_score
            for i in range(self.strategy_cfg.S_PROP_ITER):
                Q_score += (wgt_topk * Q_score[topk]).sum(-1) * self.strategy_cfg.S_PROP_COEF

            m_idxs = Q_score.sort(descending=True)[1]

            # oversample and find centroids
            M = self.strategy_cfg.S_M
            m_topk = m_idxs[:n * (1 + M)]
            km = KMeans(n_clusters=n)
            km.fit(all_embs[m_topk])
            dists = euclidean_distances(km.cluster_centers_, all_embs[m_topk])
            dists = torch.from_numpy(dists)

            m_idxs = m_idxs + acc_count
            acc_count = acc_count + len(m_idxs)
            total_m_idxs.append(m_idxs)
            total_dists.append(dists)
        total_dists = torch.cat(total_dists, dim=1).numpy()
        total_m_idxs = torch.cat(total_m_idxs, dim=0).numpy()
        print(total_dists.shape)
        print(total_m_idxs.shape)



        #
        # all_probs, all_embs, _, _ = self.predict_probs_and_embed(unlabel_loader)
        #
        # # get Q_score
        # sim = all_embs.cpu().mm(all_embs.transpose(1, 0))
        # K = self.strategy_cfg.S_K
        # sim_topk, topk = torch.topk(sim, k=K + 1, dim=1)
        # sim_topk, topk = sim_topk[:, 1:], topk[:, 1:]
        # wgt_topk = (sim_topk / sim_topk.sum(dim=1, keepdim=True))
        #
        # Q_score = -((all_probs[topk] * all_probs.unsqueeze(1)).sum(-1) * wgt_topk).sum(-1)
        #
        # # propagate Q_score
        # for i in range(self.strategy_cfg.S_PROP_ITER):
        #     Q_score += (wgt_topk * Q_score[topk]).sum(-1) * self.strategy_cfg.S_PROP_COEF
        #
        # m_idxs = Q_score.sort(descending=True)[1]
        #
        # # oversample and find centroids
        # M = self.strategy_cfg.S_M
        # m_topk = m_idxs[:n * (1 + M)]
        # km = KMeans(n_clusters=n)
        # km.fit(all_embs[m_topk])
        # dists = euclidean_distances(km.cluster_centers_, all_embs[m_topk])
        sort_idxs = total_dists.argsort(axis=1)
        q_idxs = []
        ax, rem = 0, n
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))
            rem = n - len(q_idxs)
            ax += 1
        q_idxs = total_m_idxs[q_idxs]
        # self.query_dset.rand_transform = None

        return self.label_info.unlabel_ind[q_idxs]

