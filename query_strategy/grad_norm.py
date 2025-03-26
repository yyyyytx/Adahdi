from .base_strategy import BaseStrategy
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import random

class GradNormSampling(BaseStrategy):
    '''
    Boosting Active Learning via Improving Test Performance*, AAAI 2022.
    '''
    def query(self, n):
        self.net = self.net[0]

        random.shuffle(self.label_info.unlabel_ind)
        if self.strategy_cfg.is_subset is True:
            subset = self.label_info.unlabel_ind[:self.strategy_cfg.subset]
        else:
            subset = self.label_info.unlabel_ind
        # unlabel_loader, unlabel_ind = self.build_unlabel_loader(subset)
        # uncertainty = self.grad_uncertainty(self.net, unlabel_loader)

        loader_list = self.build_divided_unlabel_loader(self.active_cfg.sub_num)
        uncertainty = []
        for i, unlabel_loader in enumerate(loader_list):
            print('unlabel loader: %d/%d' % (i + 1, len(loader_list)))
            tmp_uncertainty = self.grad_uncertainty(self.net, unlabel_loader)
            uncertainty.append(tmp_uncertainty)
        uncertainty = torch.cat(uncertainty, dim=0)
        arg = np.argsort(uncertainty)
        return subset[arg][-n:]


    def build_unlabel_loader(self, ind):
        # unlabel_ind = self.label_info.unlabel_ind
        subdataset = torch.utils.data.Subset(self.select_ds, ind)
        select_loader = DataLoader(subdataset,
                                   batch_size=1,
                                   num_workers=4,
                                   shuffle=False)
        return select_loader, ind


    def build_divided_unlabel_loader(self, loader_len):
        list_len = math.ceil(len(self.label_info.unlabel_ind)/loader_len)
        loader_list = []
        for i in range(list_len):
            if i == list_len-1:
                unlabel_ind = self.label_info.unlabel_ind[i * loader_len:]
            else:
                unlabel_ind = self.label_info.unlabel_ind[i*loader_len:(i+1)*loader_len]

            subdataset = torch.utils.data.Subset(self.select_ds, unlabel_ind)
            select_loader = DataLoader(subdataset,
                                       batch_size=1,
                                       num_workers=4,
                                       shuffle=False)
            loader_list.append(select_loader)
        return loader_list

    def grad_uncertainty(self, model, unlabeled_loader):
        model.eval()
        uncertainty = torch.tensor([]).cuda()

        criterion = nn.CrossEntropyLoss()

        for j in range(1):
            for (inputs, labels, _, _) in tqdm(unlabeled_loader):
                inputs = inputs.cuda()

                scores, _ = model(inputs)
                posterior = F.softmax(scores, dim=1)

                loss = 0.0

                posterior = posterior.squeeze()

                for i in range(self.net.n_label):
                    label = torch.full([1], i)
                    label = label.cuda()
                    loss += posterior[i] * criterion(scores, label)



                pred_gradnorm = self.compute_gradnorm(model, loss)
                pred_gradnorm = torch.sum(pred_gradnorm)
                pred_gradnorm = pred_gradnorm.unsqueeze(0)

                uncertainty = torch.cat((uncertainty, pred_gradnorm), 0)

        return uncertainty.cpu()

    def compute_gradnorm(self, model, loss):
        grad_norm = torch.tensor([]).cuda()
        gradnorm = 0.0

        model.zero_grad()
        loss.backward(retain_graph=True)
        for param in model.parameters():
            if param.grad is not None:
                gradnorm = torch.norm(param.grad)
                gradnorm = gradnorm.unsqueeze(0)
                grad_norm = torch.cat((grad_norm, gradnorm), 0)

        return grad_norm
