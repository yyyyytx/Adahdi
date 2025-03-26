import torch

from .base_strategy import BaseStrategy
import numpy as np
from sklearn.metrics import pairwise_distances
from utils import *


# class MetricSampling(BaseStrategy):
#
#     def query(self, n):
#         self.net = self.net[0]
#
#         # label_loader, label_ind = self.build_labeled_loader()
#         unlabel_loader, unlabel_ind = self.build_unlabel_loader()
#         u_probs, u_embedding, _, _ = self.predict_probs_and_embed(unlabel_loader)
#
#         mean_embedding = self.trainer.center_margin_loss.mean_feats.detach().cpu()
#
#         # distanc_matrix = pairwise_distances(u_embedding, self.mean_embedding, metric='l2')
#         distanc_matrix = torch.cdist(u_embedding, mean_embedding)
#         metric_dist = torch.min(distanc_matrix, dim=1).values
#         aa = self.strategy_cfg.gamma1
#         if self.active_cfg.rd > 3:
#             aa = self.strategy_cfg.gamma2
#         # tmp_ind = np.argsort(metric_dist)[-aa*n:]
#         tmp_ind = torch.sort(metric_dist, descending=True).indices[:aa*n]
#         query = kCenterGreedy(features=u_embedding[tmp_ind])
#         select_ind = query.select_batch_(already_selected=[], N=n)
#         query_ind = unlabel_ind[tmp_ind][select_ind]
#         return query_ind

class MetricSampling(BaseStrategy):

    def query(self, n):
        self.net = self.net[0]

        label_loader, label_ind = self.build_labeled_loader()
        l_probs, l_embedding, _, _ = self.predict_probs_and_embed(label_loader)




        unlabel_loader, unlabel_ind = self.build_unlabel_loader()
        u_probs, u_embedding, _, _ = self.predict_probs_and_embed(unlabel_loader)

        mean_embedding = self.trainer.center_margin_loss.mean_feats.detach().cpu()

        l_distanc_matrix = torch.cdist(l_embedding, mean_embedding)
        l_metric_dist = torch.min(l_distanc_matrix, dim=1).values
        l_flag = l_metric_dist > self.trainer.train_cfg.positive_margin
        print(torch.sum(l_flag))
        l_metric_dist = l_metric_dist[l_flag]
        l_embedding = l_embedding[l_flag]


        # distanc_matrix = pairwise_distances(u_embedding, self.mean_embedding, metric='l2')
        distanc_matrix = torch.cdist(u_embedding, mean_embedding)
        metric_dist = torch.min(distanc_matrix, dim=1).values
        u_flag = metric_dist > self.trainer.train_cfg.positive_margin
        if (torch.sum(u_flag) > n):
            unlabel_ind = unlabel_ind[u_flag]
            metric_dist = metric_dist[u_flag]
            u_embedding = u_embedding[u_flag]
        else:
            print('bu gou')



        aa = self.strategy_cfg.gamma1
        tmp_ind = torch.sort(metric_dist, descending=True).indices[:aa*n]

        sub_u_embedding = u_embedding[tmp_ind]
        select_embedding = torch.cat([l_embedding, sub_u_embedding], dim=0)
        print(l_embedding.shape, sub_u_embedding.shape, select_embedding.shape)

        query = kCenterGreedy(features=select_embedding)
        select_ind = query.select_batch_(already_selected=range(0, len(l_embedding)), N=n)
        select_ind = [element - len(l_embedding) for element in select_ind]
        query_ind = unlabel_ind[tmp_ind][select_ind]
        return query_ind
