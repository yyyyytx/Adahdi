import torch

from .base_strategy import BaseStrategy
import numpy as np
from sklearn.metrics import pairwise_distances
from utils import *
import math

class DUCSampling(BaseStrategy):

    def query(self, n):
        self.net = self.net[0]
        first_stat = list()

        total_distributional_uncertainty = []
        total_data_uncertainty = []
        # label_loader, label_ind = self.build_labeled_loader()
        unlabel_loader, unlabel_ind = self.build_unlabel_loader()
        with torch.no_grad():
            idx = 0
            for tgt_img, tgt_lbl, _, _ in unlabel_loader:
                tgt_img, tgt_lbl = tgt_img.cuda(), tgt_lbl.cuda()

                tgt_out, _ = self.net(tgt_img)
                tgt_out = tgt_out.cpu()

                alpha = torch.exp(tgt_out)
                total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
                expected_p = alpha / total_alpha
                eps = 1e-7

                # distributional uncertainty of each sample
                point_entropy = - torch.sum(expected_p * torch.log(expected_p + eps), dim=1)
                data_uncertainty = torch.sum(
                    (alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
                distributional_uncertainty = point_entropy - data_uncertainty
                total_distributional_uncertainty.append(distributional_uncertainty)
                total_data_uncertainty.append(data_uncertainty)

        sample_num = math.ceil(n)
        fisrt_round_num = math.ceil(n * 10)

        total_distributional_uncertainty = torch.cat(total_distributional_uncertainty, dim=0)
        total_data_uncertainty = torch.cat(total_data_uncertainty, dim=0)

        print(total_data_uncertainty.shape, total_distributional_uncertainty.shape)
        first_ind = torch.argsort(total_distributional_uncertainty, descending=True)[:fisrt_round_num]
        total_distributional_uncertainty = total_distributional_uncertainty[first_ind]
        total_data_uncertainty = total_data_uncertainty[first_ind]

        second_ind = torch.argsort(total_data_uncertainty, descending=True)[:sample_num]


        query_ind = unlabel_ind[second_ind]
        return query_ind
