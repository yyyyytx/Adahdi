from .base_trainer import BaseTrainer
import torch
from tqdm import tqdm
from trainer.losses import CenterSeperateMarginLoss
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import time
from torch import nn
from .losses import *
from torch.utils.data import DataLoader
import torch.optim as optim

class TQSTrainer(BaseTrainer):

    def __init__(self, net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg=None, is_amp=False, logger=None):
        super().__init__(net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg, is_amp, logger)
        self.net = net[0]
        self.multi_classify = net[1]
        self.discrim = net[2]
        self.multi_train_ds = train_cfg.multi_train_ds
        self.label_info.tmp_label_ind = copy.deepcopy(self.label_info.label_ind)

    def build_multi_train_label_loader(self):
        loaders = []
        for train_ds in self.multi_train_ds:
            subdataset = torch.utils.data.Subset(copy.deepcopy(train_ds), self.label_info.tmp_label_ind)
            train_loader = DataLoader(dataset=subdataset,
                                      batch_size=self.train_cfg.train_bs,
                                      num_workers=2,
                                      shuffle=True)
            loaders.append(train_loader)
        return loaders

    def build_tqs_optimizer(self):
        optimizer1 = optim.Adadelta(self.multi_classify.parameters(), lr=0.1)
        return optimizer1

    def train(self, name, n_epoch=None):
        opti, lr_sched = self.build_optimizer(self.net)
        optimizer1 = self.build_tqs_optimizer()
        train_loader = self.build_train_label_loader()
        multi_train_loaders = self.build_multi_train_label_loader()
        train_unlabel_loader = self.build_train_unlabel_loader()
        acclist = []
        best_acc = 0.
        if n_epoch == None:
            epochs = self.train_cfg.epochs
        else:
            epochs = n_epoch
        for epoch in tqdm(range(epochs)):
            accTrain, losses = self.train_each_epoch(self.net, opti, train_loader, epoch, name)
            # self.writer.add_scalar('training_accuracy/%s' % name, accTrain, epoch)
            self.train_multi_each_epoch(self.net, self.multi_classify, multi_train_loaders, optimizer1)
            self.train_sim_each_epoch(self.net, self.discrim, train_loader, train_unlabel_loader, opti)
            acclist.append(accTrain)
            if lr_sched is not None:
                lr_sched.step(epoch)

            if (epoch + 1) % self.train_cfg.val_interval == 0:
                current_acc = self.base_model_accuracy()['current']
                cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                str=cur_time + 'train epoch: %d  train acc %.4f cur acc %.4f loss: %.4f' % (epoch + 1, accTrain, current_acc, losses)
                print(str)
                self.logger.info(str)
                self.writer.add_scalar('test_accuracy/%s' % name, current_acc, epoch)
                if current_acc > best_acc:
                    best_acc = current_acc
        return best_acc

    def train_multi_each_epoch(self, net, multi_classify, multi_loaders, optimizer):
        net.eval()
        multi_classify.train()
        iters = zip(multi_loaders[0], multi_loaders[1], multi_loaders[2], multi_loaders[3], multi_loaders[4])
        for batch_idx, ((data1, target1, _,_), (data2, target2, _,_), (data3, target3, _,_),
                        (data4, target4, _,_), (data5, target5, _,_)) in enumerate(iters):
            data1 = data1.cuda()
            data2 = data2.cuda()
            data3 = data3.cuda()
            data4 = data4.cuda()
            data5 = data5.cuda()

            target1 = target1.cuda()
            target2 = target2.cuda()
            target3 = target3.cuda()
            target4 = target4.cuda()
            target5 = target5.cuda()

            with torch.no_grad():
                output1, feature1 = net(data1)
                output2, feature2 = net(data2)
                output3, feature3 = net(data3)
                output4, feature4 = net(data4)
                output5, feature5 = net(data5)

            optimizer.zero_grad()

            y1_d1, y2_d1, y3_d1, y4_s1, y5_s1 = multi_classify(feature1.detach())
            y1_d2, y2_d2, y3_d2, y4_s2, y5_s2 = multi_classify(feature2.detach())
            y1_d3, y2_d3, y3_d3, y4_s3, y5_s3 = multi_classify(feature3.detach())
            y1_d4, y2_d4, y3_d4, y4_s4, y5_s4 = multi_classify(feature4.detach())
            y1_d5, y2_d5, y3_d5, y4_s5, y5_s5 = multi_classify(feature5.detach())

            loss1 = F.cross_entropy(y1_d1, target1)
            loss2 = F.cross_entropy(y2_d2, target2)
            loss3 = F.cross_entropy(y3_d3, target3)
            loss4 = F.cross_entropy(y4_s4, target4)
            loss5 = F.cross_entropy(y5_s5, target5)
            loss = loss1 + loss2 + loss3 + loss4 + loss5

            loss.backward()
            optimizer.step()

    def train_sim_each_epoch(self, model, discrim, source_train_loader, target_train_loader, optimizer):
        model.eval()
        discrim.train()
        for batch_idx, (
        (source_data, source_label, _,_), (target_data, target_label, _,_)) in enumerate(
                zip(source_train_loader, target_train_loader)):
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data, target_label = target_data.cuda(), target_label.cuda()

            optimizer.zero_grad()

            with torch.no_grad():
                source_output, source_feature = model(source_data)
                target_output, target_feature = model(target_data)

            source_sim = discrim(source_feature.detach())
            target_sim = discrim(target_feature.detach())

            sim_loss = F.binary_cross_entropy(source_sim, torch.zeros_like(source_sim)) + \
                       F.binary_cross_entropy(target_sim, torch.ones_like(target_sim))
            sim_loss.backward()
            optimizer.step()

    # def train(self, name, n_epoch=None):