from .base_strategy import BaseStrategy
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import random
import torch.optim as optim
import torch
import math
from tqdm import tqdm
class ActiveFTSampling(BaseStrategy):
    '''
    (CVPR2023)Active Finetuning: Exploiting Annotation Budget in the Pretraining-Finetuning Paradigm
    '''

    def query(self, n):
        self.net = self.net[0]
        unlabel_loader, unlabel_ind = self.build_unlabel_loader()
        _, unlabel_embedding, _, _ = self.predict_probs_and_embed(unlabel_loader)

        unlabel_embedding = F.normalize(unlabel_embedding, dim=1)
        self.features = unlabel_embedding.cuda()
        self.total_num = len(unlabel_ind)
        self.slice = self.total_num
        sample_ids = list(range(self.total_num))
        initial_idx = random.sample(sample_ids, n)
        centroids = self.features[initial_idx]
        self.centroids = nn.Parameter(centroids).cuda()
        self.sample_num = n
        optimizer = optim.Adam(self.parameters(), lr=self.strategy_cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.strategy_cfg.max_iter, eta_min=1e-6)

        for i in tqdm(range(self.strategy_cfg.max_iter)):
            loss = self.get_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        centroids = self.centroids.detach()
        centroids = F.normalize(centroids, dim=1)
        slice = 100
        sample_slice_num = math.ceil(centroids.shape[0] / slice)
        sample_ids = set()
        # _, ids_sort = torch.sort(dist, dim=1, descending=True)
        for sid in range(sample_slice_num):
            start = sid * slice
            end = min((sid + 1) * slice, centroids.shape[0])
            dist = torch.matmul(centroids[start:end], self.features.transpose(1, 0))  # (slice_num, n)
            _, ids_sort = torch.sort(dist, dim=1, descending=True)
            for i in range(ids_sort.shape[0]):
                for j in range(ids_sort.shape[1]):
                    if ids_sort[i, j].item() not in sample_ids:
                        sample_ids.add(ids_sort[i, j].item())
                        break

        print(len(sample_ids))
        sample_ids = list(sample_ids)
        return unlabel_ind[sample_ids]

    def get_loss(self):
        centroids = F.normalize(self.centroids, dim=1)
        prod = torch.matmul(self.features, centroids.transpose(1, 0))  # (n, k)
        prod = prod / self.strategy_cfg.temperature
        prod_exp = torch.exp(prod)
        prod_exp_pos, pos_k = torch.max(prod_exp, dim=1)  # (n, )

        cent_prod = torch.matmul(centroids.detach(), centroids.transpose(1, 0))  # (k, k)
        cent_prod = cent_prod / self.strategy_cfg.temperature
        cent_prod_exp = torch.exp(cent_prod)
        cent_prob_exp_sum = torch.sum(cent_prod_exp, dim=0)  # (k, )

        J = torch.log(prod_exp_pos) - torch.log(prod_exp_pos + cent_prob_exp_sum[pos_k])
        J = -torch.mean(J)

        return J
