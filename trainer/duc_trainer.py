from .base_trainer import BaseTrainer
import torch
from tqdm import tqdm
from torch.autograd import grad
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, MultiStepLR
import torch.optim as optim
from models.tqs_module import ReverseLayerF
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from itertools import repeat
import copy
from itertools import  cycle
from torch.utils.data.sampler import RandomSampler, BatchSampler
from trainer.losses import *
class DUCTrainer(BaseTrainer):

    def __init__(self, net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg=None, is_amp=False, logger=None):
        super().__init__(net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg, is_amp, logger)
        self.net = net[0]



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
            if lr_sched is not None:
                lr_sched.step(epoch)
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

    def combine_train_each_epoch(self, optimizer, train_loader, epoch, name):
        self.net.train()
        accFinal, tot_loss, iters = 0., 0., 0

        net_opit = optimizer

        l_loader = train_loader[0]
        u_loader = train_loader[1]
        src_sup_wt, lambda_unsup, lambda_cent = 1.0, 0.1, 0.1
        l_loader = iter(l_loader)
        joint_loader = zip(l_loader, u_loader)

        edl_criterion = EDL_Loss()

        for batch_idx, ((Xs_l, ys_l, _, ds_ind_l), (Xs_u, _, _, ds_ind_u)) in enumerate(joint_loader):
            total_loss = 0
            Xs_l, ys_l, ds_ind_l = Xs_l.cuda(), ys_l.cuda(), ds_ind_l.cuda()
            Xs_u, ds_ind_u = Xs_u.cuda(), ds_ind_u.cuda()
            with torch.enable_grad():
                y_hats, l_feats = self.net(Xs_l)
                Loss_nll_s, Loss_KL_s = edl_criterion(y_hats, ys_l)
                Loss_KL_s = Loss_KL_s / self.net.n_label

                total_loss += Loss_nll_s
                total_loss += Loss_KL_s


                tgt_unlabeled_out, _ = self.net(Xs_u)
                alpha_t = torch.exp(tgt_unlabeled_out)
                total_alpha_t = torch.sum(alpha_t, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
                expected_p_t = alpha_t / total_alpha_t
                eps = 1e-7
                point_entropy_t = - torch.sum(expected_p_t * torch.log(expected_p_t + eps), dim=1)
                data_uncertainty_t = torch.sum(
                    (alpha_t / total_alpha_t) * (torch.digamma(total_alpha_t + 1) - torch.digamma(alpha_t + 1)), dim=1)
                loss_Udis = torch.sum(point_entropy_t - data_uncertainty_t) / tgt_unlabeled_out.shape[0]
                loss_Udata = torch.sum(data_uncertainty_t) / tgt_unlabeled_out.shape[0]

                total_loss += 1.0 * loss_Udis
                total_loss += 0.05 * loss_Udata


                net_opit.zero_grad()
                total_loss.backward()
                net_opit.step()
        iters += 1
        tot_loss += total_loss
        # print('epoch %d training loss %.4f %.4f'%(epoch, loss_ct, loss_ce))
        self.writer.add_scalar('training_loss/%s' % name, tot_loss / iters, epoch)

        return accFinal / len(self.label_info.label_ind), tot_loss / iters

        # for Xs, ys, ind, ds_ind in train_loader:
        #     Xs = Xs.cuda()
        #     ys = ys.cuda()
        #     ds_ind = ds_ind.cuda()
        #     with torch.enable_grad():
        #         y_hats, feats = self.net(Xs)
        #         disc_input = ReverseLayerF.apply(feats, self.train_cfg.alpha)
        #         disc_out = self.domain_discriminator(disc_input)
        #         disc_loss = F.cross_entropy(disc_out, ds_ind)
        #         classifier_loss = F.cross_entropy(y_hats, ys)
        #         print('dis loss %.4f, cls loss %.4f' % (disc_loss, classifier_loss))
        #         total_losses = disc_loss + classifier_loss
        #
        #     accFinal += torch.sum((torch.max(y_hats, 1)[1] == ys).float()).data.item()
        #
        #     net_opit.zero_grad()
        #     dis_opit.zero_grad()
        #     total_losses.backward()
        #     net_opit.step()
        #     dis_opit.step()
        #
        #
        # iters += 1
        # tot_loss += total_losses
        # # print('epoch %d training loss %.4f %.4f'%(epoch, loss_ct, loss_ce))
        # self.writer.add_scalar('training_loss/%s' % name, tot_loss / iters, epoch)
        #
        # return accFinal / len(self.label_info.label_ind), tot_loss / iters
        #
        #
