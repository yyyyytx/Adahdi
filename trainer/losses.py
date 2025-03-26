import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy

class MultiCenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, ds_count=1 , use_gpu=True):
        super(MultiCenterLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        self.ds_count = ds_count
        # self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).cuda())

        if use_gpu:
            self.centers = nn.Parameter(torch.zeros(ds_count, self.num_class, self.num_feature).cuda())
        else:
            self.centers = nn.Parameter(torch.zeros(ds_count, self.num_class, self.num_feature))

    def initial_centers(self, centers):
        assert self.centers.shape == centers.shape
        self.centers.data = centers

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def forward(self, x, labels, idx, logger):
        x = F.normalize(x, dim=1)
        center = self.centers[idx]
        ind = labels.repeat((self.centers.shape[2], 1)).T
        ind = torch.unsqueeze(ind, dim=1)
        center = torch.gather(center, dim=1, index=ind)
        center = torch.squeeze(center)
        center = F.normalize(center, dim=1)

        cos_sim = (x * center).sum(dim=-1)
        dist = 1. - cos_sim
        ct_loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        var_loss = 0.0
        for c in range(self.num_class):
            cls_centers = F.normalize(self.centers[:,c,:], dim=1)
            mean_cls_centers = cls_centers.mean(dim=0).repeat((len(cls_centers), 1))
            cls_cos_sim = (cls_centers * mean_cls_centers).sum(dim=-1)
            cls_sim_var = (1. - cls_cos_sim).mean(dim=-1)
            var_loss += cls_sim_var
        var_loss = var_loss / self.num_class

        return ct_loss + var_loss


class MultiCenterMarginLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, ds_count=1, pos_margin=0.5, centers_margin=2.0, use_gpu=True):
        super(MultiCenterMarginLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        self.ds_count = ds_count
        self.pos_margin = pos_margin
        self.centers_margin = centers_margin
        if use_gpu:
            self.centers = nn.Parameter(torch.zeros(ds_count, self.num_class, self.num_feature).cuda())
        else:
            self.centers = nn.Parameter(torch.zeros(ds_count, self.num_class, self.num_feature))
        self.ema_decay = 0.999

    def initial_centers(self, centers):
        assert self.centers.shape == centers.shape
        self.centers.data = centers


    def forward(self, x, labels, idx, logger):
        x = F.normalize(x, dim=1)
        loss = 0.
        for ds_ind in range(len(self.centers)):
            ds_centers = F.normalize(self.centers[ds_ind], dim=1)
            ds_mask = idx == ds_ind
            dist = (x[ds_mask] - ds_centers[labels[ds_mask]]).pow(2).sum(dim=-1).sqrt()
            print(dist)
            loss += torch.sum(torch.clamp(dist - self.pos_margin, min=1e-12))
        loss = loss / len(labels)



        # center = self.centers[idx]
        # ind = labels.repeat((self.centers.shape[2], 1)).T
        # ind = torch.unsqueeze(ind, dim=1)
        # center = torch.gather(center, dim=1, index=ind)
        # center = torch.squeeze(center)
        # center = F.normalize(center, dim=1)
        # dist = (x - center).pow(2).sum(dim=-1).sqrt()
        # loss = torch.clamp(dist - self.pos_margin, min=1e-12).mean(dim=-1)

        norm_centers = []
        for ds_ind in range(len(self.centers)):
            ds_centers = F.normalize(self.centers[ds_ind], dim=1)
            norm_centers.append(torch.unsqueeze(ds_centers, dim=0))
        norm_centers = torch.cat(norm_centers, dim=0)
        mean = torch.sum(norm_centers, dim=0).repeat((self.ds_count, 1, 1)).detach() / self.ds_count
        var = torch.pow(norm_centers - mean, 2).sum() / self.ds_count

        ds_margin_losses =0.
        #
        #
        for ds_ind in range(len(self.centers)):
            ds_centers = F.normalize(self.centers[ds_ind], dim=1)
            # ds_labels = labels[idx == ds_ind]
            # ds_feats = x[idx == ds_ind]
            ds_dist = torch.cdist(ds_centers, ds_centers, compute_mode='donot_use_mm_for_euclid_dist')
            # print(torch.eye(len(ds_dist)))
            margin_dist = torch.clamp(self.centers_margin - ds_dist - torch.eye(len(ds_dist)).cuda() * self.centers_margin, min=0.)
            ds_loss1 = torch.sum(margin_dist) / (ds_dist.shape[0]*(ds_dist.shape[1]-1))
            ds_margin_losses += ds_loss1
            # ds_margin_losses += torch.sum(ds_loss1)
            # ds_loss1 = torch.sum(ds_dist)/(ds_dist.shape[0]*(ds_dist.shape[1]-1))
            # ds_margin_losses +=torch.clamp(self.centers_margin - ds_loss1, min=0.)
        #
        ds_margin_losses = ds_margin_losses / len(self.centers)



        str = 'center loss %.4f var %.4f margin %.4f' % (loss, var, ds_margin_losses)
        print(str)
        logger.info(str)
        return loss + var + ds_margin_losses

    def ema_update(self, feats, labels, ds_idx):
        mean_features = torch.zeros(self.ds_count, self.num_class, self.num_feature).cuda()
        feats = F.normalize(feats, dim=1)
        for ds_ind in range(self.ds_count):
            ds_feats = feats[ds_idx == ds_ind]
            ds_labels = labels[ds_idx == ds_ind]

            for c in range(self.num_class):
                f = ds_feats[ds_labels == c]
                if len(f) != 0:
                    mean_features[ds_ind][c] = torch.mean(f, dim=0)

        # ema mean feat
        self.centers.data = self.centers.data * self.ema_decay + (1 - self.ema_decay) * mean_features



class EMAMultiCenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, ds_count=1 , use_gpu=True):
        super(EMAMultiCenterLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        self.ds_count = ds_count
        self.centers = torch.zeros(ds_count, self.num_class, self.num_feature).cuda()
        self.ema_decay = 0.999


    def initial_centers(self, centers):
        assert self.centers.shape == centers.shape
        self.centers.data = centers

    def ema_update(self, feats, labels, ds_idx):
        mean_features = torch.zeros(self.ds_count, self.num_class, self.num_feature).cuda()
        feats = F.normalize(feats, dim=1)
        for ds_ind in range(self.ds_count):
            ds_feats = feats[ds_idx == ds_ind]
            ds_labels = labels[ds_idx == ds_ind]

            for c in range(self.num_class):
                f = ds_feats[ds_labels == c]
                if len(f) != 0:
                    mean_features[ds_ind][c] = torch.mean(f, dim=0)

        # ema mean feat
        self.centers = self.centers * self.ema_decay + (1 - self.ema_decay) * mean_features



    def forward(self, x, labels, idx, logger):
        x = F.normalize(x, dim=1)
        center = self.centers[idx].detach()
        ind = labels.repeat((self.centers.shape[2], 1)).T
        ind = torch.unsqueeze(ind, dim=1)
        center = torch.gather(center, dim=1, index=ind)
        center = torch.squeeze(center)
        center = F.normalize(center, dim=1)
        dist = (x - center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        norm_centers = []
        for ds_ind in range(len(self.centers)):
            ds_centers = F.normalize(self.centers[ds_ind], dim=1)
            norm_centers.append(torch.unsqueeze(ds_centers, dim=0))
        norm_centers = torch.cat(norm_centers, dim=0)
        mean = torch.sum(norm_centers, dim=0).repeat((self.ds_count, 1, 1)).detach() / self.ds_count
        var = torch.pow(norm_centers - mean, 2).sum() / self.ds_count
        str = 'center loss %.4f var %.4f' % (loss, var)
        print(str)
        # ds_margin_losses =0.
        #
        #
        # for ds_ind in range(len(self.centers)):
        #     ds_centers = F.normalize(self.centers[ds_ind], dim=1)
        #     ds_labels = labels[idx == ds_ind]
        #     ds_feats = x[idx == ds_ind]
        #     ds_dist = torch.cdist(ds_feats, ds_centers, compute_mode='donot_use_mm_for_euclid_dist')
        #
        #     onehot_label = F.one_hot(ds_labels, ds_dist.shape[1])
        #     # ds_loss = torch.sum(onehot_label * ds_dist,1).unsqueeze(1) * torch.ones_like(ds_dist) - ds_dist
        #     ds_delta_dist = ds_dist - torch.sum(onehot_label * ds_dist,1).unsqueeze(1) * torch.ones_like(ds_dist)
        #     ds_loss1 = torch.sum(torch.clamp(1.0 - ds_delta_dist - onehot_label, min=0.))/(ds_dist.shape[0]*ds_dist.shape[1])
        #     ds_margin_losses +=ds_loss1
        #
        # ds_margin_losses = ds_margin_losses / len(self.centers)
            # ds_centers_dist = torch.cdist(ds_centers, ds_centers, compute_mode='donot_use_mm_for_euclid_dist') + torch.eye(len(ds_centers)).cuda()
            # ds_centers_dist = torch.cdist(ds_centers, ds_centers, compute_mode='donot_use_mm_for_euclid_dist') + torch.eye(len(ds_centers)).cuda()
            # ds_loss = torch.sum(torch.clamp_min(1.0 - ds_centers_dist, min=0.))
            # ds_margin_losses +=ds_loss
        # onehot_label = F.one_hot(labels, x.shape[1])
        # loss_m = torch.sum(torch.clamp(1.0 - (torch.sum(onehot_label * x,1).unsqueeze(1) * torch.ones_like(x) - x)\
        # -onehot_label,min=0.0))/(x.shape[0]*x.shape[1])
        # cross_data_var = self.centers
        # str = 'center loss %.4f  margin %.4f' % (loss, ds_margin_losses)
        # print(str)
        # logger.info(str)
        return loss #+ ds_margin_losses

# def CenterMarginLoss(feat, logit, label, margin=1., keepdim=False):
#     onehot_label = F.one_hot(label, intensor.shape[1])
#     weight = torch.ones_like(intensor)
#
#     assert weight.shape == intensor.shape
#
#     if not keepdim:
#         loss = torch.sum(weight * \
#         torch.clamp(margin - (torch.sum(onehot_label * intensor,1).unsqueeze(1) * torch.ones_like(intensor) - intensor)\
#         -onehot_label,min=0.0))/(intensor.shape[0]*intensor.shape[1])
#     else:
#         loss = torch.sum(weight * \
#         torch.clamp(margin - (torch.sum(onehot_label * intensor,1).unsqueeze(1) * torch.ones_like(intensor) - intensor)\
#         -onehot_label,min=0.0),1,keepdim=True)/(intensor.shape[0]*intensor.shape[1])
#

class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).cuda())

    def forward(self, x, labels, idx):
        x = F.normalize(x, dim=1)
        self.centers = F.normalize(self.centers, dim=1)

        center = self.centers[labels]
        dist = (x - center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        # ds_centers = F.normalize(self.centers[ds_ind], dim=1)
        # ds_labels = labels[idx == ds_ind]
        # ds_feats = x[idx == ds_ind]
        # ds_dist = torch.cdist(ds_feats, ds_centers, compute_mode='donot_use_mm_for_euclid_dist')
        #
        # onehot_label = F.one_hot(ds_labels, ds_dist.shape[1])
        # # ds_loss = torch.sum(onehot_label * ds_dist,1).unsqueeze(1) * torch.ones_like(ds_dist) - ds_dist
        # ds_delta_dist = ds_dist - torch.sum(onehot_label * ds_dist, 1).unsqueeze(1) * torch.ones_like(ds_dist)
        # ds_loss1 = torch.sum(torch.clamp(1.0 - ds_delta_dist - onehot_label, min=0.)) / (
        #             ds_dist.shape[0] * ds_dist.shape[1])
        # ds_margin_losses += ds_loss1

        return loss

    def initial_centers(self, centers):
        assert self.centers.shape == centers.shape
        self.centers.data = centers

class CenterCosLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterCosLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        if use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def forward(self, x, labels):
        center = self.centers[labels]
        norm_c = self.l2_norm(center)
        norm_x = self.l2_norm(x)
        similarity = (norm_c * norm_x).sum(dim=-1)
        dist = 1.0 - similarity
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss

from typing import Tuple

import torch
from torch import nn, Tensor

def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss



def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss


class CenterSeperateMarginLoss(nn.Module):
    def __init__(self,
                 in_feats = 2,
                 n_classes = 10,
                 margin = 0.25,
                 distance = 1.,
                 initial_center = None):
        super(CenterSeperateMarginLoss, self).__init__()

        if initial_center is None:
            self.mean_feats = torch.zeros(n_classes, in_feats)
        else:
            self.mean_feats = torch.tensor(initial_center)
        self.old_mean_feats = None
        self.margin = margin
        self.distance = distance


        # self.ema_decay = 0.999
        self.ema_decay = 0.999
        self.ema_iteration = 0

        self.n_classes = n_classes
        self.in_feats = in_feats

    def forward(self, x, labels):

        self.ema_mean(x.detach(), labels)
        self.ema_iteration += 1

        delta = torch.cdist(x, self.mean_feats.cuda().detach())

        positive_mask = (torch.arange(self.n_classes).expand(len(x), -1).cuda() == labels.expand(self.n_classes, -1).transpose(0,1)).float()
        negative_mask = 1. - positive_mask


        ps = torch.clamp((delta - self.margin), min=0.) * positive_mask
        ns = torch.clamp((self.distance - delta), min=0.) * negative_mask

        ap = torch.clamp_min(ps.detach() + self.distance, min=0.) * positive_mask
        an = torch.clamp_min(ns.detach() + self.margin , min=0. ) * negative_mask

        # prevent divide zero
        loss_p = torch.sum(ap * ps) / (torch.sum(ps > 0.) + 1)
        loss_n = torch.sum(an * ns) / (torch.sum(ns > 0.) + 1)

        return torch.log(1 + loss_n + loss_p)


    def ema_mean(self, feats, labels):
        mean_features = torch.zeros(self.n_classes, self.in_feats)
        for c in range(self.n_classes):
            f = feats[labels==c]
            if len(f) != 0:
                mean_features[c] = torch.mean(feats[labels==c], dim=0)
        # ema mean feat
        alpha = min(1 - 1 / (self.ema_iteration + 1), self.ema_decay)


        if self.old_mean_feats is None:
            self.mean_feats = mean_features
            self.old_mean_feats = mean_features
        else:
            self.mean_feats = self.old_mean_feats * alpha + (1 - alpha) * mean_features
            # self.old_mean_feats = mean_features
            self.old_mean_feats = self.mean_feats

def Margin_loss(intensor,label,margin=1,keepdim=False,weight=None):

    onehot_label = F.one_hot(label,intensor.shape[1])
    weight = torch.ones_like(intensor)

    assert weight.shape == intensor.shape

    if not keepdim:
        loss = torch.sum(weight * \
        torch.clamp(margin - (torch.sum(onehot_label * intensor,1).unsqueeze(1) * torch.ones_like(intensor) - intensor)\
        -onehot_label,min=0.0))/(intensor.shape[0]*intensor.shape[1])
    else:
        loss = torch.sum(weight * \
        torch.clamp(margin - (torch.sum(onehot_label * intensor,1).unsqueeze(1) * torch.ones_like(intensor) - intensor)\
        -onehot_label,min=0.0),1,keepdim=True)/(intensor.shape[0]*intensor.shape[1])

    return loss

class NLLLoss(nn.Module):
    """
    NLL loss for energy based model
    """

    def __init__(self, cfg):
        super(NLLLoss, self).__init__()
        assert cfg.ENERGY_BETA > 0, "beta for energy calculate must be larger than 0"
        self.beta = cfg.ENERGY_BETA

    def forward(self, inputs, targets):
        indices = torch.unsqueeze(targets, dim=1)
        energy_c = torch.gather(inputs, dim=1, index=indices)

        all_energy = -1.0 * self.beta * inputs
        free_energy = -1.0 * torch.logsumexp(all_energy, dim=1, keepdim=True) / self.beta

        nLL = energy_c - free_energy

        return nLL.mean()


def bound_max_loss(energy, bound):
    """
    return the loss value of max(0, \mathcal{F}(x) - \Delta )
    """
    energy_minus_bound = energy - bound
    energy_minus_bound = torch.unsqueeze(energy_minus_bound, dim=1)
    zeros = torch.zeros_like(energy_minus_bound)
    for_select = torch.cat((energy_minus_bound, zeros), dim=1)
    selected = torch.max(for_select, dim=1).values

    return selected.mean()

class FreeEnergyAlignmentLoss(nn.Module):
    """
    free energy alignment loss
    """

    def __init__(self, cfg):
        super(FreeEnergyAlignmentLoss, self).__init__()
        assert cfg.ENERGY_BETA > 0, "beta for energy calculate must be larger than 0"
        self.beta = cfg.ENERGY_BETA

        self.type = cfg.ENERGY_ALIGN_TYPE

        if self.type == 'l1':
            self.loss = nn.L1Loss()
        elif self.type == 'mse':
            self.loss = nn.MSELoss()
        elif self.type == 'max':
            self.loss = bound_max_loss

    def forward(self, inputs, bound):
        mul_neg_beta = -1.0 * self.beta * inputs
        log_sum_exp = torch.logsumexp(mul_neg_beta, dim=1)
        free_energies = -1.0 * log_sum_exp / self.beta

        bound = torch.ones_like(free_energies) * bound
        loss = self.loss(free_energies, bound)

        return loss


class ConditionalEntropyLoss(nn.Module):

    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)

class EDL_Loss(nn.Module):
    """
    evidence deep learning loss
    """
    def __init__(self):
        super(EDL_Loss, self).__init__()


    def forward(self, logits, labels=None):
        alpha = torch.exp(logits)
        total_alpha = torch.sum(alpha, dim=1, keepdim=True)
        if labels is None:
            labels = torch.max(alpha, dim=1)[1]

        one_hot_y = torch.eye(logits.shape[1]).cuda()
        one_hot_y = one_hot_y[labels]
        one_hot_y.requires_grad = False
        loss_nll = torch.sum(one_hot_y * (total_alpha.log() - alpha.log())) / logits.shape[0]
        uniform_bata = torch.ones((1, logits.shape[1])).cuda()
        uniform_bata.requires_grad = False
        total_uniform_beta = torch.sum(uniform_bata, dim=1)
        new_alpha = one_hot_y + (1.0 - one_hot_y) * alpha
        new_total_alpha = torch.sum(new_alpha, dim=1)
        loss_KL = torch.sum(
            torch.lgamma(new_total_alpha) - torch.lgamma(total_uniform_beta) - torch.sum(torch.lgamma(new_alpha), dim=1) \
            + torch.sum((new_alpha - 1) * (torch.digamma(new_alpha) - torch.digamma(new_total_alpha.unsqueeze(1))), dim=1)
        ) / logits.shape[0]

        return loss_nll, loss_KL
