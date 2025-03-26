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
from torch import nn


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class MCCTrainer(BaseTrainer):

    def __init__(self, net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg=None, is_amp=False, logger=None):
        super().__init__(net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg, is_amp, logger)
        self.net = net[0]



    def train(self, name, n_epoch=None):

        opti, lr_sched = self.build_optimizer(self.net)

        train_loader = self.build_train_label_loader()
        unlabeled_loader = self.build_train_unlabel_loader()
        acclist = []
        best_acc = 0.

        if n_epoch == None:
            epochs = self.train_cfg.epochs
        else:
            epochs = n_epoch

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



        for batch_idx, ((Xs_l, ys_l, _, ds_ind_l), (Xs_u, _, _, ds_ind_u)) in enumerate(joint_loader):
            Xs_l, ys_l, ds_ind_l = Xs_l.cuda(), ys_l.cuda(), ds_ind_l.cuda()
            Xs_u, ds_ind_u = Xs_u.cuda(), ds_ind_u.cuda()
            with torch.enable_grad():
                y_hats, l_feats = self.net(Xs_l)
                classifier_loss = F.cross_entropy(y_hats, ys_l)

                output_tu, _ = self.net(Xs_u)

                outputs_target_temp = output_tu / 0.05
                target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
                target_entropy_weight = Entropy(target_softmax_out_temp).detach()
                target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
                target_entropy_weight = 64 * target_entropy_weight / torch.sum(
                    target_entropy_weight)
                cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(
                    target_softmax_out_temp)
                cov_matrix_t = cov_matrix_t / (torch.sum(cov_matrix_t, dim=1) + 1e-12)
                mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / self.net.n_label

                # total_feats = torch.cat((l_feats, u_feats), 0)
                # total_ds_labels = torch.cat((ds_ind_l, ds_ind_u), 0)
                # disc_input = ReverseLayerF.apply(total_feats, self.train_cfg.alpha)
                # disc_out = self.domain_discriminator(disc_input)
                # disc_loss = F.cross_entropy(disc_out, total_ds_labels)
                # print('dis loss %.4f, cls loss %.4f' % (disc_loss, classifier_loss))
                total_losses = mcc_loss + classifier_loss

                net_opit.zero_grad()
                total_losses.backward()
                net_opit.step()
        iters += 1
        tot_loss += total_losses
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
