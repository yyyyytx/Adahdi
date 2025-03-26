from .base_trainer import BaseTrainer
import torch
from tqdm import tqdm
from trainer.losses import CenterSeperateMarginLoss
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import time
from torch import nn
from .losses import *

class SDMTrainer(BaseTrainer):

    def amp_train(self, net, scaler, optimizer, Xs, ys):
        optimizer.zero_grad()
        with autocast():
            logit, feats = net(Xs)
            loss = F.cross_entropy(logit, ys)
            label = torch.unsqueeze(ys,dim=1)
            onehot_label = torch.zeros_like(logit).scatter_(1,label.long(),1)
            addition_loss = (F.normalize(logit) * onehot_label).sum() / len(Xs)
            smooth_marginloss = 1 - (torch.sum(onehot_label * logit,1).unsqueeze(1) * torch.ones_like(logit) - logit)
            margin_loss = Margin_loss(F.normalize(logit), ys, weight=smooth_marginloss)
            total_loss = margin_loss - 1 * addition_loss + loss
            # total_loss = loss


        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss, logit

    def no_amp_train(self, net, optimizer, Xs, ys):
        optimizer.zero_grad()
        with torch.enable_grad():
            logit, feats = net(Xs)
            loss = F.cross_entropy(logit, ys)
            label = torch.unsqueeze(ys, dim=1)
            onehot_label = torch.zeros_like(logit).scatter_(1, label.long(), 1)
            addition_loss = (F.normalize(logit) * onehot_label).sum() / len(Xs)
            smooth_marginloss = 1 - (torch.sum(onehot_label * logit, 1).unsqueeze(1) * torch.ones_like(logit) - logit)
            margin_loss = Margin_loss(F.normalize(logit), ys, weight=smooth_marginloss)
            total_loss = margin_loss - 1 * addition_loss + loss
        total_loss.backward()
        optimizer.step()
        return loss, logit
