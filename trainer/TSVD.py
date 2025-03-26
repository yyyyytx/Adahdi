from .base_trainer import BaseTrainer
import torch
from tqdm import tqdm
from torch.autograd import grad
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, MultiStepLR
import torch.optim as optim
from .losses import CenterLoss, MultiCenterLoss, Margin_loss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from itertools import repeat
import copy
from itertools import  cycle
# def repeater(data_loader):
#     for loader in repeat(data_loader):
#         for data in loader:
#             yield data

class TSVDTrainer(BaseTrainer):

    def __init__(self, net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg=None, is_amp=False, logger=None):
        super().__init__(net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg, is_amp, logger)
        self.U = self.net[1]
        self.net = self.net[0]


    def train(self, name, n_epoch=None):
        opti, lr_sched = self.build_optimizer(self.net)
        train_loader = self.build_train_label_loader()
        unlabeled_loader = self.build_train_unlabel_loader()

        acclist = []
        best_acc = 0.
        for epoch in tqdm(range(self.train_cfg.epochs)):
            accTrain, lossTrain = self.combine_train_each_epoch(opti, [train_loader, unlabeled_loader], epoch, name)

            self.writer.add_scalar('training_accuracy/%s' % name, accTrain, epoch)
            acclist.append(accTrain)
            lr_sched.step()

            if (epoch + 1) % self.train_cfg.val_interval == 0:
                result = self.base_model_accuracy()
                print(result)
                current_acc = result['current']
                cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                str = cur_time + 'train epoch: %d  train acc %.4f cur acc %.4f loss %.4f' % (epoch + 1, accTrain, current_acc, lossTrain)
                print(str)
                self.logger.info(str)
                self.writer.add_scalar('test_accuracy/%s' % name, current_acc, epoch)
                if current_acc > best_acc:
                    best_acc = current_acc

        return best_acc

    def pseudo_label(self, logit, feat, log_var):
        pred = F.softmax(logit, dim=1)
        entropy = (-pred * torch.log(pred)).sum(-1)
        label = torch.argmax(logit, dim=-1).long()

        mask = (entropy < self.args.entropy_thr).float()
        index = torch.nonzero(mask).squeeze(-1)
        feat_ = torch.index_select(feat, 0, index)
        label_ = torch.index_select(label, 0, index)
        log_var_ = torch.index_select(log_var, 0, index)

        return feat_, label_, log_var_


    def euclid_dist(self, x, y):
        x_sq = (x ** 2).mean(-1)
        x_sq_ = torch.stack([x_sq] * y.size(0), dim = 1)
        y_sq = (y ** 2).mean(-1)
        y_sq_ = torch.stack([y_sq] * x.size(0), dim = 0)
        xy = torch.mm(x, y.t()) / x.size(-1)
        dist = x_sq_ + y_sq_ - 2 * xy
        return dist

    def prototype_align(self, logits):
        KL_loss = 0
        criterion_KL = nn.KLDivLoss()
        criterion_MSE = nn.MSELoss(size_average=True)
        for i in range(self.ndomain):
            for j in range(i, self.ndomain):
                KL_loss += criterion_KL(logits[i].log(), logits[j]) + criterion_KL(logits[j].log(), logits[i])
                KL_loss += criterion_MSE(self.mean[i], self.mean[j])
        return KL_loss

    # update prototypes and adjacency matrix
    def update_statistics(self, feats, labels, epsilon=1e-5):
        num_labels = 0
        loss_local = 0

        for domain_idx in range(self.ndomain):
            tmp_feat = feats[domain_idx]
            tmp_label = labels[domain_idx]
            num_labels += tmp_label.shape[0]

            if tmp_label.shape[0] == 0:
                break
                # tmp_mean = torch.zeros((self.args.nclasses, self.args.nfeat)).cuda()
            else:
                onehot_label = torch.zeros((tmp_label.shape[0], self.args.nclasses)).scatter_(1, tmp_label.unsqueeze(
                    -1).cpu(), 1).float().cuda()
                domain_feature = tmp_feat.unsqueeze(1) * onehot_label.unsqueeze(-1)
                tmp_mean = domain_feature.sum(0) / (onehot_label.unsqueeze(-1).sum(0) + epsilon)

                tmp_mask = (tmp_mean.sum(-1) != 0).float().unsqueeze(-1)
                self.mean[domain_idx] = self.mean[domain_idx].detach() * (1 - tmp_mask) + (
                        self.mean[domain_idx].detach() * self.args.beta + tmp_mean * (1 - self.args.beta)) * tmp_mask

                tmp_dist = self.euclid_dist(self.mean[domain_idx], self.mean[domain_idx])
                self.adj[domain_idx] = torch.exp(-tmp_dist / (2 * self.args.sigma ** 2))

                domain_feature_center = onehot_label.unsqueeze(-1) * self.mean[domain_idx].unsqueeze(0)
                tmp_mean_center = domain_feature_center.sum(1)
                # compute local relation alignment loss
                loss_local += (((tmp_mean_center - tmp_feat) ** 2).mean(-1)).sum()

        return self.adj, loss_local / num_labels


    def combine_train_each_epoch(self, optimizer, train_loader, epoch, name):
        self.net.train()
        accFinal, tot_loss, iters = 0., 0., 0

        l_loader = train_loader[0]
        u_loader = train_loader[1]
        net_opit = optimizer


        l_loader = iter(l_loader)
        joint_loader = zip(l_loader, u_loader)


        for batch_idx, ((Xs_l, ys_l, _, ds_ind_l), (Xs_u, _, _, ds_ind_u)) in enumerate(joint_loader):
            Xs_l, ys_l, ds_ind_l = Xs_l.cuda(), ys_l.cuda(), ds_ind_l.cuda()
            Xs_u, ds_ind_u = Xs_u.cuda(), ds_ind_u.cuda()
            img_s = list()
            label_s = list()
            with torch.enable_grad():
                for i in range(self.train_cfg.dataset_number):
                    tmp_img = Xs_l[ds_ind_l == i]
                    tmp_label = ys_l[ds_ind_l == i]
                    img_s.append(tmp_img)
                    label_s.append(tmp_label)

                feat_s = list()
                logit_s = list()
                log_var_s = list()
                for domain_idx in range(self.train_cfg.dataset_number):
                    tmp_img = img_s[domain_idx]
                    tmp_logit, tmp_feat = self.net(tmp_img)
                    tmp_feat = F.normalize(tmp_feat, p=2, dim=1)
                    feat_s.append(tmp_feat)
                    logit_s.append(tmp_logit)
                    tmp_var = self.U(tmp_feat)
                    log_var_s.append(tmp_var)

                img_t = Xs_u[ds_ind_u==self.train_cfg.dataset_number]
                if (len(img_t) != 0):
                    _, feat_t = self.net(img_t)
                    feat_t = F.normalize(feat_t, p=2, dim=1)
                    logit_t = self.C(feat_t)
                    log_var_t = self.C(feat_t).U(feat_t)
                    feat_t_, label_t_, log_var_t_ = self.pseudo_label(logit_t, feat_t, log_var_t)
                    feat_s.append(feat_t_)
                    label_s.append(label_t_)
                    log_var_s.append(log_var_t_)

                feat_var = list()
                for domain_idx in range(self.train_cfg.dataset_number):
                    feat_var.append(feat_s[domain_idx])
                feat_var.append(feat_s[domain_idx + 1])
                self.adj, loss_local = self.update_statistics(feat_var, label_s)


        for Xs, ys, ind, ds_ind in train_loader:
            Xs = Xs.cuda()
            ys = ys.cuda()
            img_s = list()
            label_s = list()
            with torch.enable_grad():
                for i in range(self.train_cfg.dataset_number):
                    tmp_img = Xs[ds_ind==i]
                    tmp_label = ys[ds_ind==i]
                    img_s.append(tmp_img)
                    label_s.append(tmp_label)

            feat_s = list()



            Xs = Xs.cuda()
            ys = ys.cuda()
            ds_ind = ds_ind.cuda()
            with torch.enable_grad():
                y_hats, feats = self.net(Xs)



                # print('loss %.4f %.4f %.4f' % (loss_ce, 0., 0.))
                total_losses = loss_ce + loss_ct + margin_loss

            accFinal += torch.sum((torch.max(y_hats, 1)[1] == ys).float()).data.item()

            net_opit.zero_grad()
            center_opit.zero_grad()
            total_losses.backward()
            net_opit.step()
            for param in self.center_loss.parameters():
                param.grad.data *= (1./(ct_weight + 1e-12))
            center_opit.step()

        iters += 1
        tot_loss += total_losses
        # print('epoch %d training loss %.4f %.4f'%(epoch, loss_ct, loss_ce))
        self.writer.add_scalar('training_loss/%s' % name, tot_loss / iters, epoch)

        return accFinal / len(self.label_info.label_ind), tot_loss / iters





