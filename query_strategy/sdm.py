from .base_strategy import BaseStrategy
import numpy as np
import time
import torch
from copy import deepcopy
import torch.nn.functional as F


class SDMSampling(BaseStrategy):
    '''
    (CVPR2022)Learning Distinctive Margin toward Active Domain Adaptation
    '''

    def query(self, n_select):
        self.net = self.trainer.net

        # unlabel_loader, unlabel_ind = self.build_unlabel_loader()
        loader_list = self.build_divided_unlabel_loader(self.active_cfg.sub_num)
        st_uncertainty = []
        st_probs = []
        for i, unlabel_loader in enumerate(loader_list):
            print('unlabel loader: %d/%d' % (i+1, len(loader_list)))
            tmp_uncertainty, tmp_probs = self.pdf(unlabel_loader)
            print(tmp_uncertainty.shape, tmp_probs.shape)
            st_uncertainty.append(tmp_uncertainty)
            st_probs.append(tmp_probs)
        st_uncertainty = torch.cat(st_uncertainty, dim=0)
        st_probs = torch.cat(st_probs, dim=0)
        print(st_uncertainty.shape, st_probs.shape)

        # st_uncertainty, st_probs = self.pdf(unlabel_loader)
        s_probs_sorted, _ = st_probs.sort(descending=True)
        margin = s_probs_sorted[:, 0] - s_probs_sorted[:, 1]
        uncertainty = margin - self.strategy_cfg.SDM_LAMBDA * st_uncertainty
        uncertainty = uncertainty.sort()[1].numpy()
        chosen = uncertainty[:n_select]
        return self.label_info.unlabel_ind[chosen]


    def pdf(self, unlabel_loader):
        s_probs, s_embs, _, _ = self.predict_probs_and_embed(unlabel_loader)
        s_weight = self.net.fc.weight.cpu()
        start_time = time.time()
        s_probs_sorted, s_pos_sorted = s_probs.sort(descending=True)
        s_pos_weight = s_weight[s_pos_sorted]
        s_pmax1 = s_probs_sorted[:,0].reshape(s_pos_sorted.shape[0],1)
        s_pmax2 = s_probs_sorted[:,1].reshape(s_pos_sorted.shape[0],1)
        s_max1 = (s_pmax1 * (1-s_pmax1)) * s_pos_weight[:,0,:]
        s_max2 = (s_pmax2 * (1-s_pmax2)) * s_pos_weight[:,1,:]
        s_Q_tensor = s_max1 - s_max2 \
        - torch.sum((s_probs_sorted[:,2:].unsqueeze(-1) * s_pos_weight[:,2:,:]),dim=1) \
        * ((s_probs_sorted[:, 0] - s_probs_sorted[:, 1]).unsqueeze(1))
        # Loss pdf
        s_embs = s_embs.requires_grad_()
        fc = deepcopy(self.net.fc).cpu()
        s_logits = fc(s_embs)
        max1_pseudo = s_pos_sorted[:,0]
        max2_pseudo = s_pos_sorted[:,1]
        ce_loss1 = F.cross_entropy(s_logits, max1_pseudo)
        ce_loss2 = F.cross_entropy(s_logits, max2_pseudo)
        margin_loss1 = F.multi_margin_loss(F.normalize(s_logits),max1_pseudo)
        margin_loss2 = F.multi_margin_loss(F.normalize(s_logits),max2_pseudo)
        label1 = torch.unsqueeze(max1_pseudo,dim=1)
        label2 = torch.unsqueeze(max2_pseudo,dim=1)
        onehot_label1 = torch.zeros_like(s_logits).scatter_(1,label1.long(),1)
        onehot_label2 = torch.zeros_like(s_logits).scatter_(1,label2.long(),1)
        addition_loss1 = (F.normalize(s_logits) * onehot_label1).sum() / s_pos_sorted.shape[0]
        addition_loss2 = (F.normalize(s_logits) * onehot_label2).sum() / s_pos_sorted.shape[0]
        loss1 = ce_loss1 + margin_loss1 - addition_loss1
        loss2 = ce_loss2 + margin_loss2 - addition_loss2
        grad1 = -torch.autograd.grad(outputs=loss1,inputs=s_embs,retain_graph=True)[0]
        grad2 = -torch.autograd.grad(outputs=loss2,inputs=s_embs,retain_graph=True)[0]
        grad = s_probs_sorted[:,0].unsqueeze(-1) * grad1 + s_probs_sorted[:,1].unsqueeze(-1) * grad2
        # uncertainty
        uncertainty = torch.cosine_similarity(s_Q_tensor,grad)
        return uncertainty, s_probs
