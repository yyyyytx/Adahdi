from .base_strategy import BaseStrategy
import numpy as np
import time
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn


class TQSSampling(BaseStrategy):
    '''
    (ICCV2021)TQS
    '''

    def query(self, n_select):
        label_loader, label_ind = self.build_labeled_loader()
        unlabel_loader, unlabel_ind = self.build_unlabel_loader()

        model = self.net[0]
        multi = self.net[1]
        discrim = self.net[2]
        uncertainty_rank = self.uncertainty_evaluate(model, multi, discrim, unlabel_loader)
        test_acc = self.strategy_cfg.cur_acc
        sub_count = round(test_acc.item() * len(unlabel_ind))
        if sub_count > n_select:
            tmp_select_ind = torch.argsort(uncertainty_rank, descending=True)[:sub_count].cpu().numpy()
        else:
            tmp_select_ind = torch.argsort(uncertainty_rank, descending=True)[:n_select].cpu()
        select_ind = tmp_select_ind[np.random.permutation(len(tmp_select_ind))[:n_select]]


        # np.random.choice()

        return unlabel_ind[select_ind]

    def uncertainty_evaluate(self, model, multi, discrim, unlabel_loader):
        model.eval()
        multi.eval()
        stat = list()
        uncertainty_list = []
        with torch.no_grad():
            for batch_idx, (data, target, _,_) in enumerate(unlabel_loader):
                data, target = data.cuda(), target.cuda()
                output, feature = model(data)
                y1, y2, y3, y4, y5 = multi(feature)
                target_sim = discrim(feature.detach())

                # uncertainty = margin(output) + get_consistency(y1, y2, y3, y4, y5) + target_sim
                uncertainty = margin(output) + get_consistency(y1, y2, y3, y4, y5)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred))
                uncertainty_list.append(uncertainty)
        uncertainty_list = torch.cat(uncertainty_list, dim=0).cpu()

        return uncertainty_list

def get_consistency(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5):
    fc2_s = nn.Softmax(-1)(fc2_s)
    fc2_s2 = nn.Softmax(-1)(fc2_s2)
    fc2_s3 = nn.Softmax(-1)(fc2_s3)
    fc2_s4 = nn.Softmax(-1)(fc2_s4)
    fc2_s5 = nn.Softmax(-1)(fc2_s5)

    fc2_s = torch.unsqueeze(fc2_s, 1)
    fc2_s2 = torch.unsqueeze(fc2_s2, 1)
    fc2_s3 = torch.unsqueeze(fc2_s3, 1)
    fc2_s4 = torch.unsqueeze(fc2_s4, 1)
    fc2_s5 = torch.unsqueeze(fc2_s5, 1)
    c = torch.cat((fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5), dim=1)
    d = torch.std(c, 1)
    consistency = torch.mean(d, 1)
    return consistency

def margin(out):
    out = nn.Softmax(-1)(out)
    top2 = torch.topk(out, 2).values
    # print(top2)
    return 1 - (top2[:, 0] - top2[:, 1])