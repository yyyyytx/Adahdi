from .base_trainer import BaseTrainer
import torch
from tqdm import tqdm
from trainer.losses import CenterSeperateMarginLoss
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import time

class MetricTrainer(BaseTrainer):
    def __init__(self, net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg=None, is_amp=False, logger=None):
        super().__init__(net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg, is_amp, logger)
        self.net = net[0]

    def train(self, name, n_epoch=None):
        train_loader = self.build_train_label_loader()
        self.center_margin_loss = CenterSeperateMarginLoss(in_feats=self.net.get_embedding_dim(),
                                                           n_classes=self.net.n_label,
                                                           margin=self.train_cfg.positive_margin,
                                                           distance=self.train_cfg.distance)

        opti, lr_sched = self.build_optimizer(self.net)
        acclist = []
        best_acc = 0.
        for epoch in tqdm(range(self.train_cfg.epochs)):
            accTrain, lossTrain = self.metric_train_each_epoch(opti, train_loader, epoch, name)
            self.writer.add_scalar('training_accuracy/%s' % name, accTrain, epoch)
            acclist.append(accTrain)
            lr_sched.step(epoch)

            if (epoch + 1) % self.train_cfg.val_interval == 0:
                current_acc = self.base_model_accuracy()['total']
                cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                str = cur_time + 'train epoch: %d  train acc %.4f cur acc %.4f loss %.4f' % (
                epoch + 1, accTrain, current_acc, lossTrain)
                print(str)
                self.logger.info(str)
                self.writer.add_scalar('test_accuracy/%s' % name, current_acc, epoch)
                if current_acc > best_acc:
                    best_acc = current_acc
        return best_acc

    def metric_train_each_epoch(self, optimizer, train_loader, epoch, name):
        self.net.train()
        accFinal, tot_loss, iters = 0., 0., 0
        scaler = GradScaler()

        for Xs, ys, ind , _ in train_loader:
            Xs = Xs.cuda()
            ys = ys.cuda()
            if self.is_amp == True:
                with autocast():
                    y_hats, feats = self.net(Xs)
                    if epoch >= self.train_cfg.metric_epoch:
                        loss_ct = self.center_margin_loss(feats, ys) * self.train_cfg.lam
                        loss_ce = F.cross_entropy(y_hats, ys)
                        # print('loss %.4f %.4f' % (loss_ce, loss_ct / self.train_cfg.lam))
                        total_losses = loss_ce + loss_ct
                    else:
                        loss_ce = F.cross_entropy(y_hats, ys)
                        total_losses = loss_ce
                    accFinal += torch.sum((torch.max(y_hats, 1)[1] == ys).float()).data.item()
                    optimizer.zero_grad()
                    scaler.scale(total_losses).backward()
                    scaler.step(optimizer)
                    scaler.update()

            else:
                with torch.enable_grad():
                    y_hats, feats = self.net(Xs)
                    if epoch >= self.train_cfg.metric_epoch:
                        loss_ct = self.center_margin_loss(feats, ys) * self.train_cfg.lam
                        loss_ce = F.cross_entropy(y_hats, ys)
                        # print('loss %.4f %.4f' % (loss_ce, loss_ct / self.train_cfg.lam))
                        total_losses = loss_ce + loss_ct
                    else:
                        loss_ce = F.cross_entropy(y_hats, ys)
                        total_losses = loss_ce
                    accFinal += torch.sum((torch.max(y_hats, 1)[1] == ys).float()).data.item()
                    optimizer.zero_grad()
                    total_losses.backward()
                    optimizer.step()
                    # scaler.update()
                # raise NotImplementedError

        iters+=1
        tot_loss += total_losses
        return accFinal / len(self.label_info.label_ind), tot_loss / iters



