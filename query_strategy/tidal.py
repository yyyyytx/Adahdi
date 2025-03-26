import torch

from .base_strategy import BaseStrategy
import numpy as np
from sklearn.metrics import pairwise_distances
from utils import *
import torch.nn.functional as F


class TiDALSampling(BaseStrategy):

    def query(self, n):
        self.pred_module = self.net[1]
        self.net = self.net[0]
        unlabel_loader, unlabel_ind = self.build_unlabel_loader()

        loader_list = self.build_divided_unlabel_loader(self.active_cfg.sub_num)
        uncertainty_all = []
        for i, unlabel_loader in enumerate(loader_list):
            print('unlabel loader: %d/%d' % (i + 1, len(loader_list)))
            uncertainty = self.tidal_predict(unlabel_loader)
            uncertainty_all.append(uncertainty)
        uncertainty_all = torch.cat(uncertainty_all, dim=0)

        # uncertainty = self.tidal_predict(unlabel_loader)
        uncertainty_ind = torch.argsort(uncertainty_all, descending=True)[:n].cpu()

        return unlabel_ind[uncertainty_ind]


    def tidal_predict(self, unlabel_loader):
        self.net.eval()
        self.pred_module.eval()

        sub_logit_all = []

        with torch.no_grad():
            for x, y, ind, _, _ in unlabel_loader:
                inputs = x.cuda()
                scores, emb, features = self.net.forward_features(inputs)

                sub_logit = self.pred_module(features)
                sub_logit = sub_logit.detach().cpu()
                sub_logit_all.append(sub_logit)
        sub_logit_all = torch.cat(sub_logit_all, dim=0).cpu()
        sub_prob = torch.softmax(sub_logit_all, dim=1)

        sub_entropy = -(sub_prob * torch.log(sub_prob)).sum(dim=1)  # Entropy
        uncertainty = sub_entropy
        return uncertainty