from .base_strategy import BaseStrategy
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import torch
import torch.nn as nn

class CLUESampling(BaseStrategy):

    '''
    (ICCV2021)Active Domain Adaptation via Clustering UNcertainty-weighted Embeddings
    '''
    def query(self, n_select):
        self.net = self.net[0]

        unlabel_loader, unlabel_ind = self.build_unlabel_loader()
        loader_list = self.build_divided_unlabel_loader(self.active_cfg.sub_num)
        total_dists = []

        for i, unlabel_loader in enumerate(loader_list):
            print('unlabel loader: %d/%d' % (i + 1, len(loader_list)))


            tgt_scores, embeddings, _, logits = self.predict_probs_and_embed(unlabel_loader)

            tgt_pen_emb = embeddings.cpu().numpy()
            # tgt_scores = nn.Softmax(dim=1)(logits / self.T)
            tgt_scores += 1e-8
            sample_weights = -(tgt_scores * torch.log(tgt_scores)).sum(1).cpu().numpy()

            # Run weighted K-means over embeddings
            km = KMeans(n_select)
            km.fit(tgt_pen_emb, sample_weight=sample_weights)

            # Find nearest neighbors to inferred centroids
            dists = euclidean_distances(km.cluster_centers_, tgt_pen_emb)
            dists = torch.from_numpy(dists)
            total_dists.append(dists)
        total_dists = torch.cat(total_dists, dim=1).numpy()
        sort_idxs = total_dists.argsort(axis=1)
        q_idxs = []
        ax, rem = 0, n_select
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))
            rem = n_select - len(q_idxs)
            ax += 1
        return unlabel_ind[q_idxs]