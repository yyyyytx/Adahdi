import random

from .base_strategy import BaseStrategy
import numpy as np
from sklearn.metrics import pairwise_distances
from utils import *
from torch.utils.data import DataLoader


class LLALSampling(BaseStrategy):

    def query(self, n):
        # self.loss_module = self.net[1]
        # self.net = self.net[0]
        self.loss_module = self.trainer.loss_module
        self.net = self.trainer.net

        random.shuffle(self.label_info.unlabel_ind)
        if self.strategy_cfg.is_subset is True:
            subset = self.label_info.unlabel_ind[:self.strategy_cfg.subset]
        else:
            subset = self.label_info.unlabel_ind

        # subset = self.label_info.unlabel_ind[:self.strategy_cfg.subset]
        unlabel_loader, unlabel_ind = self.build_unlabel_loader(subset)
        uncertainty = self.get_uncertainty(self.net, unlabel_loader)
        # Index in ascending order
        arg = np.argsort(uncertainty)
        return subset[arg][-n:]



    def build_unlabel_loader(self, ind):
        subdataset = torch.utils.data.Subset(self.select_ds, ind)
        select_loader = DataLoader(subdataset,
                                   batch_size=self.active_cfg.select_bs,
                                   num_workers=4,
                                   shuffle=False)
        return select_loader, ind

    def get_uncertainty(self, models, unlabeled_loader):
        self.net.eval()
        self.loss_module.eval()
        uncertainty = torch.tensor([]).cuda()

        with torch.no_grad():
            for Xs, ys, ind , _ in unlabeled_loader:
                Xs, ys = Xs.cuda(), ys.cuda()
                scores, embd, features = self.net.forward_features(Xs)
                pred_loss = self.loss_module(features)  # pred_loss = criterion(scores, labels) # ground truth loss
                pred_loss = pred_loss.view(pred_loss.size(0))
                uncertainty = torch.cat((uncertainty, pred_loss), 0)
        return uncertainty.cpu()
