from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from trainer import *
import math
import torch.nn as nn
import numpy as np
import copy

class BaseStrategy(nn.Module):
    def __init__(self, net, active_cfg, strategy_cfg, label_info, select_ds, trainer):
        super().__init__()
        self.net = net
        self.select_ds = select_ds
        self.active_cfg = active_cfg
        self.label_info = label_info
        self.strategy_cfg = strategy_cfg
        # self.writer = writer
        self.query_count = 0
        self.trainer = trainer
        # self.trainer = BaseTrainer(net, train_params, train_ds, test_ds, writer)


    # def start_train(self, name):
    #     self.trainer.train(name)

    def build_unlabel_loader(self):
        unlabel_ind = self.label_info.unlabel_ind
        subdataset = torch.utils.data.Subset(self.select_ds, unlabel_ind)
        select_loader = DataLoader(subdataset,
                                   batch_size=self.active_cfg.select_bs,
                                   num_workers=4,
                                   shuffle=False)
        return select_loader, unlabel_ind

    def build_sequence_unlabel_loader(self):
        sorted_label_ind = np.sort(self.label_info.unlabel_ind)
        # subdataset = torch.utils.data.Subset(copy.deepcopy(self.train_ds), sorted_label_ind)
        subdataset = torch.utils.data.Subset(self.select_ds, sorted_label_ind)
        select_loader = DataLoader(subdataset,
                                   batch_size=self.active_cfg.select_bs,
                                   num_workers=4,
                                   shuffle=False)
        return select_loader, sorted_label_ind

    def build_ds_sequence_train_label_loader(self):
        sorted_label_ind = np.sort(self.label_info.label_ind)
        subdataset = torch.utils.data.Subset(copy.deepcopy(self.train_ds), sorted_label_ind)
        train_loader = DataLoader(dataset=subdataset,
                                  batch_size=self.train_cfg.train_bs,
                                  num_workers=4,
                                  shuffle=False)
        return train_loader

    def build_labeled_loader(self):
        label_ind = self.label_info.label_ind
        subdataset = torch.utils.data.Subset(self.select_ds, label_ind)
        select_loader = DataLoader(subdataset,
                                   batch_size=self.active_cfg.select_bs,
                                   num_workers=4,
                                   shuffle=False)
        return select_loader, label_ind

    def build_divided_unlabel_loader(self, loader_len, shuffle=False):
        # loader_len = len(self.label_info.unlabel_ind) // divided_count
        # list_len = math.ceil(len(self.label_info.unlabel_ind)/loader_len)
        list_len = math.floor(len(self.label_info.unlabel_ind) / loader_len)
        loader_list = []
        if shuffle == True:
            unlabel_inds = torch.randperm(len(self.label_info.unlabel_ind)).numpy()
        else:
            unlabel_inds = np.arange(len(self.label_info.unlabel_ind))

        if list_len == 0:
            unlabel_ind = self.label_info.unlabel_ind
            subdataset = torch.utils.data.Subset(self.select_ds, unlabel_ind)
            select_loader = DataLoader(subdataset,
                                       batch_size=self.active_cfg.select_bs,
                                       num_workers=4,
                                       shuffle=shuffle)
            loader_list.append(select_loader)
        else:
            for i in range(list_len):
                if i == list_len-1:
                    unlabel_ind = self.label_info.unlabel_ind[unlabel_inds][i * loader_len:]
                else:
                    unlabel_ind = self.label_info.unlabel_ind[unlabel_inds][i*loader_len:(i+1)*loader_len]

                subdataset = torch.utils.data.Subset(self.select_ds, unlabel_ind)
                select_loader = DataLoader(subdataset,
                                           batch_size=self.active_cfg.select_bs,
                                           num_workers=4,
                                           shuffle=shuffle)
                loader_list.append(select_loader)
        return loader_list

    def predict_probs_and_embed(self, data_loader, eval=True, isHalf=False):
        probs = []
        logits = []
        embedding_features = []
        true_labels = []

        if eval:
            self.net.eval()
        else:
            self.net.train()

        for x, y, ind, _ in data_loader:
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                out, e1 = self.net(x)
            prob = F.softmax(out, dim=1)
            # probs[idxs] = prob.cpu()
            # embeddings[idxs] = e1.cpu()

            logits.append(out)
            true_labels.append(y)
            embedding_features.append(e1)
            probs.append(prob)

        logits = torch.cat(logits, dim=0).cpu()
        probs = torch.cat(probs, dim=0).cpu()
        true_labels = torch.cat(true_labels, dim=0).cpu()
        embedding_features = torch.cat(embedding_features, dim=0).cpu()

        if isHalf is True:
            embedding_features = embedding_features.half()

        return probs, embedding_features, true_labels, logits

    def query(self, n):
        pass