from .base_trainer import BaseTrainer
import torch
from tqdm import tqdm
from trainer.losses import CenterSeperateMarginLoss
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import time
from torch import nn
from .losses import *

class EADATrainer(BaseTrainer):

    def train(self, name, n_epoch=None):
        opti = self.build_optimizer(self.net)
        train_label_loader = self.build_train_label_loader()
        train_unlabel_loader = self.build_train_unlabel_loader()
        acclist = []
        best_acc = 0.

        if n_epoch == None:
            epochs = self.train_cfg.epochs
        else:
            epochs = n_epoch
        for epoch in tqdm(range(epochs)):
            accTrain, losses = self.train_da_each_epoch(self.net, opti, train_label_loader, train_unlabel_loader, epoch, name)
            self.writer.add_scalar('training_accuracy/%s' % name, accTrain, epoch)

            acclist.append(accTrain)
            # lr_sched.step()

            if (epoch + 1) % self.train_cfg.val_interval == 0:
                current_acc = self.base_model_accuracy()['total']
                cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                str = cur_time + 'train epoch: %d  train acc %.4f cur acc %.4f loss: %.4f' % (
                epoch + 1, accTrain, current_acc, losses)
                print(str)
                self.logger.info(str)
                self.writer.add_scalar('test_accuracy/%s' % name, current_acc, epoch)
                if current_acc > best_acc:
                    best_acc = current_acc
        return best_acc

    def build_optimizer(self, net):
        import torch.optim as optim
        # create optimizer
        optimizer_cfg = self.train_cfg.optimizer

        optimizer = optim.Adadelta(net.parameters(), lr=optimizer_cfg.lr)


        return optimizer

    def train_da_each_epoch(self, net, optimizer, label_loader, unlabel_loader, epoch, name):

        # energy loss function
        nll_criterion = NLLLoss(self.strategy_cfg)

        # unsupervised energy alignment bound loss
        uns_criterion = FreeEnergyAlignmentLoss(self.strategy_cfg)

        net.train()
        accFinal, tot_loss, iters = 0., 0., 0
        true_labels = []
        scaler = GradScaler()
        total_label_len = len(label_loader.dataset)

        iter_per_epoch = max(len(label_loader), len(unlabel_loader))
        label_iter = iter(label_loader)
        unlabel_iter = iter(unlabel_loader)
        # print(len(label_loader), len(unlabel_loader))
        # print(unlabel_loader)
        for batch_idx in range(iter_per_epoch):
            try:
                label_data = label_iter.next()
            except Exception as e:
                print(e)
                print('re-gen label loader')
                label_iter = iter(label_loader)
                label_data = label_iter.next()

            try:
                unlabel_data = unlabel_iter.next()
            except Exception as e:
                print(e)
                print('re-gen unlabel loader')
                unlabel_iter = iter(unlabel_loader)
                unlabel_data = unlabel_iter.next()

            label_img, label_target = label_data[0], label_data[1]
            label_img, label_target = label_img.cuda(), label_target.cuda()

            unlabel_img, unlabel_target = unlabel_data[0], unlabel_data[1]
            unlabel_img, unlabel_target = unlabel_img.cuda(), unlabel_target.cuda()

            if self.is_amp == True:
                optimizer.zero_grad()
                total_loss = 0.
                with autocast():
                    y_hats, feats = net(label_img)
                    nll_loss = nll_criterion(y_hats, label_target)
                    total_loss += nll_loss

                    # energy alignment loss on unlabeled target data
                    unlabel_y_hats, unlabel_feats = net(unlabel_img)
                    with torch.no_grad():
                        # free energy of samples
                        output_div_t = -1.0 * self.strategy_cfg.ENERGY_BETA * y_hats
                        output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
                        free_energy = -1.0 * output_logsumexp / self.strategy_cfg.ENERGY_BETA

                        src_batch_free_energy = free_energy.mean().detach()

                        # init global mean free energy
                        if epoch == 0 and batch_idx == 0:
                            self.global_mean = src_batch_free_energy
                        # update global mean free energy
                        self.global_mean = momentum_update(self.global_mean, src_batch_free_energy)

                    fea_loss = uns_criterion(inputs=unlabel_y_hats, bound=self.global_mean)

                    total_loss += self.strategy_cfg.ENERGY_ALIGN_WEIGHT * fea_loss
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # loss, y_hats = self.no_amp_train(net, optimizer, Xs, ys)
                raise NotImplementedError

            tot_loss += total_loss.item()
            iters += 1
            accFinal += torch.sum((torch.max(y_hats, 1)[1] == label_target).float()).data.item()
        self.writer.add_scalar('training_loss/%s' % name, tot_loss / (iters + 1), epoch)
        return accFinal / total_label_len, tot_loss / (iters + 1)


def momentum_update(ema, current):
    lambd = np.random.uniform()
    return ema * lambd + current * (1 - lambd)