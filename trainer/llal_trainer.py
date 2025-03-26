from .base_trainer import BaseTrainer
import torch
from tqdm import tqdm
from trainer.losses import CenterSeperateMarginLoss
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import time
from torch import nn
from .losses import *

class LLALTrainer(BaseTrainer):
    def __init__(self, net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg=None, is_amp=False, logger=None):
        super().__init__(net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg, is_amp, logger)
        self.loss_module = net[1]
        self.net = net[0]

    def train(self, name, n_epoch=None):
        train_loader = self.build_train_label_loader()

        opti, lr_sched = self.build_optimizer(self.net)
        loss_opti, loss_sched = self.build_optimizer(self.loss_module)
        acclist = []
        best_acc = 0.
        for epoch in tqdm(range(self.train_cfg.epochs)):
            accTrain, lossTrain = self.loss_train_epoch_train([opti, loss_opti], train_loader, epoch, name)
            self.writer.add_scalar('training_accuracy/%s' % name, accTrain, epoch)
            acclist.append(accTrain)
            if lr_sched is not None:
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

    def loss_train_epoch_train(self, optimizer, train_loader, epoch, name):
        self.net.train()
        self.loss_module.train()
        accFinal, tot_loss, iters = 0., 0., 0
        net_opit = optimizer[0]
        loss_opit = optimizer[1]
        scaler = GradScaler()
        criterion = nn.CrossEntropyLoss(reduction='none')

        for Xs, ys, ind , _ in train_loader:
            Xs = Xs.cuda()
            ys = ys.cuda()


            # if self.is_amp == True:
            with torch.enable_grad():
                y_hats, embd, feats = self.net.forward_features(Xs)
                target_loss = F.cross_entropy(y_hats, ys, reduction='none')
                if epoch > self.train_cfg.epoch_loss:
                    # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
                    feats[0] = feats[0].detach()
                    feats[1] = feats[1].detach()
                    feats[2] = feats[2].detach()
                    feats[3] = feats[3].detach()
                pred_loss = self.loss_module(feats)
                pred_loss = pred_loss.view(pred_loss.size(0))

                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                m_module_loss = LossPredLoss(pred_loss, target_loss, margin=self.train_cfg.margin)
                loss = m_backbone_loss + self.train_cfg.weight * m_module_loss

            # else:
            #     raise NotImplementedError
            accFinal += torch.sum((torch.max(y_hats, 1)[1] == ys).float()).data.item()
            net_opit.zero_grad()
            loss_opit.zero_grad()
            loss.backward()
            net_opit.step()
            loss_opit.step()
        iters+=1
        tot_loss += loss
        return accFinal / len(self.label_info.label_ind), tot_loss / iters
