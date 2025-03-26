from .base_strategy import BaseStrategy
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans
import copy


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def compute_density(features, uncertainties=None, max_increase=0.1):
    normalized_uncertainty = sigmoid(uncertainties - uncertainties.mean())
    densities = []
    for i in range(features.shape[0]):
        weight = 1 + normalized_uncertainty[i] * max_increase
        distance = torch.norm(features[i] - features, dim=1) * weight
        density = 1.0 / (torch.sum(distance, dim=0).item() + 1e-7)
        densities.append(density)
    densities = torch.tensor(densities)
    return densities

class MADASampling(BaseStrategy):

    def query(self, n):
        self.net = self.net[0]
        unlabel_loader, unlabel_ind = self.build_unlabel_loader()



        total_features, total_uncertainty = self.predict_probs_and_embed(unlabel_loader)
        sub = self.active_cfg.rd * n
        sub_ind = torch.argsort(total_uncertainty, descending=True)[:sub]
        sub_uncertainty = total_uncertainty[sub_ind]
        sub_feat = total_features[sub_ind]

        densities = compute_density(sub_feat, sub_uncertainty)

        subsub_ind = torch.argsort(densities, descending=False)[:n]
        return unlabel_ind[sub_ind[subsub_ind]]


    def predict_probs_and_embed(self, data_loader, eval=True, isHalf=False):

        total_features = []
        total_uncertainty = []


        if eval:
            self.net.eval()
        else:
            self.net.train()

        for x, y, ind, _ in data_loader:
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                tgt_out , tgt_features = self.net.forward_mada(x, return_feat=True)
            alpha = torch.exp(tgt_out)
            total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
            expected_p = alpha / total_alpha
            eps = 1e-7
            point_entropy = - torch.sum(expected_p * torch.log(expected_p + eps), dim=1)
            data_uncertainty = torch.sum(
                (alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
            distributional_uncertainty = point_entropy - data_uncertainty

            final_uncertainty = self.strategy_cfg.LAMBDA_1 * distributional_uncertainty + self.strategy_cfg.LAMBDA_2 * data_uncertainty

            total_features.append(tgt_features)
            total_uncertainty.append(final_uncertainty)

        total_features = torch.cat(total_features, dim=0).cpu()
        total_uncertainty = torch.cat(total_uncertainty, dim=0).cpu()
        return total_features, total_uncertainty

        #     prob = F.softmax(out, dim=1)
        #     # probs[idxs] = prob.cpu()
        #     # embeddings[idxs] = e1.cpu()
        #
        #     logits.append(out)
        #     true_labels.append(y)
        #     embedding_features.append(e1)
        #     probs.append(prob)
        #
        # logits = torch.cat(logits, dim=0).cpu()
        # probs = torch.cat(probs, dim=0).cpu()
        # true_labels = torch.cat(true_labels, dim=0).cpu()
        # embedding_features = torch.cat(embedding_features, dim=0).cpu()
        #
        # if isHalf is True:
        #     embedding_features = embedding_features.half()
        #
        # return probs, embedding_features, true_labels, logits