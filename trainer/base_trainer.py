import torch
import torch.optim as optim
import timm
from timm.scheduler import create_scheduler
# from timm.optim import create_optimizer_v2, optimizer_kwargs
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, MultiStepLR
import time
from torch.cuda.amp import autocast, GradScaler
import torch.utils.data.distributed
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
import numpy as np


class BaseTrainer(nn.Module):
    def __init__(self, net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg=None, is_amp=False, logger=None):
        super().__init__()
        self.writer = writer
        self.net = net
        self.label_info = label_info
        self.train_cfg = train_cfg
        self.train_ds = train_ds
        self.select_ds = select_ds
        self.test_ds = test_ds
        self.is_amp = is_amp
        self.strategy_cfg = strategy_cfg
        self.logger = logger
        self.net = self.net[0]



    def build_optimizer(self, net):
        # create optimizer
        optimizer_cfg = self.train_cfg.optimizer
        if optimizer_cfg.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=optimizer_cfg.lr,
                                  momentum=optimizer_cfg.momentum,
                                  weight_decay=optimizer_cfg.weight_decay)
        elif optimizer_cfg.type == 'AdamW':
            optimizer = optim.AdamW(net.parameters(), lr=optimizer_cfg.lr)
        elif optimizer_cfg.type =='Adadelta':
            optimizer = optim.Adadelta(net.parameters(), lr=optimizer_cfg.lr)
        else:
            raise 'incorrect optimizer'

        scheduler_cfg = self.train_cfg.scheduler
        if scheduler_cfg.type == 'CosineLR':
            # lr_sched = CosineAnnealingLR(optimizer, T_max=self.train_cfg.epochs, eta_min=scheduler_cfg.eta_min)
            lr_sched = timm.scheduler.CosineLRScheduler(optimizer=optimizer,
                                                        t_initial=self.train_cfg.epochs,
                                                        lr_min=scheduler_cfg.eta_min,
                                                        warmup_t=scheduler_cfg.warmup_t)
        elif scheduler_cfg.type == 'MultiStepLR':
            lr_sched = MultiStepLR(optimizer, milestones=scheduler_cfg.milestones)
        elif scheduler_cfg.type == 'None':
            lr_sched = None
        else:
            raise 'incorrect lr sched'

        return optimizer, lr_sched

    def build_train_label_loader(self, shuffle=True):
        subdataset = torch.utils.data.Subset(copy.deepcopy(self.train_ds), self.label_info.label_ind)
        train_loader = DataLoader(dataset=subdataset,
                                  batch_size=self.train_cfg.train_bs,
                                  num_workers=2,
                                  shuffle=shuffle)
        return train_loader




    def build_ds_sequence_train_label_loader(self):
        sorted_label_ind = np.sort(self.label_info.label_ind)
        subdataset = torch.utils.data.Subset(copy.deepcopy(self.train_ds), sorted_label_ind)
        train_loader = DataLoader(dataset=subdataset,
                                  batch_size=self.train_cfg.train_bs,
                                  num_workers=4,
                                  shuffle=False)
        return train_loader

    def build_weighted_train_label_loader(self):
        subdataset = torch.utils.data.Subset(copy.deepcopy(self.train_ds), self.label_info.label_ind)
        ds_inds = torch.tensor(self.train_ds.ds_ind)[self.label_info.label_ind]
        ds_count = torch.zeros(self.label_info.l_train_ds_number)
        for i in range(self.label_info.l_train_ds_number):
            ds_count[i] = torch.sum(ds_inds==i)
        ds_weight = ds_count / torch.sum(ds_count)
        train_weight = ds_weight[ds_inds]
        # print(train_weight)

        sampler = WeightedRandomSampler(weights=train_weight, num_samples=len(train_weight))
        train_loader = DataLoader(dataset=subdataset,
                                  batch_size=self.train_cfg.train_bs,
                                  num_workers=4,
                                  sampler=sampler,
                                  shuffle=False)
        return train_loader

    def build_train_unlabel_loader(self):
        subdataset = torch.utils.data.Subset(copy.deepcopy(self.train_ds), self.label_info.unlabel_ind)
        train_loader = DataLoader(dataset=subdataset,
                                  batch_size=self.train_cfg.train_bs,
                                  num_workers=2,
                                  shuffle=True)
        return train_loader


    def build_test_loader(self):
        test_loader = DataLoader(dataset=self.test_ds,
                                 batch_size=self.train_cfg.train_bs,
                                 num_workers=4,
                                 shuffle=False)
        return test_loader


    def build_select_unlabel_loader(self):
        unlabel_ind = self.label_info.unlabel_ind
        subdataset = torch.utils.data.Subset(self.select_ds, unlabel_ind)
        select_loader = DataLoader(subdataset,
                                   batch_size=self.train_cfg.test_bs,
                                   num_workers=4,
                                   shuffle=False)
        return select_loader, unlabel_ind

    def build_select_labeled_loader(self):
        label_ind = self.label_info.label_ind
        subdataset = torch.utils.data.Subset(self.select_ds, label_ind)
        select_loader = DataLoader(subdataset,
                                   batch_size=self.train_cfg.test_bs,
                                   num_workers=4,
                                   shuffle=False)
        return select_loader, label_ind

    def build_subloader(self, dataset, sub_ind, shuffle=True):
        subdataset = torch.utils.data.Subset(dataset, sub_ind)
        sub_loader = DataLoader(subdataset,
                               batch_size=self.train_cfg.test_bs,
                               num_workers=4,
                               shuffle=shuffle)
        return sub_loader

    def build_alldata_loader(self):
        loader = DataLoader(dataset=self.select_ds,
                            batch_size=self.train_cfg.train_bs,
                            num_workers=4,
                            shuffle=False)
        return loader

    def train(self, name, n_epoch=None):
        opti, lr_sched = self.build_optimizer(self.net)
        train_loader = self.build_train_label_loader()
        acclist = []
        best_acc = 0.

        if n_epoch == None:
            epochs = self.train_cfg.epochs
        else:
            epochs = n_epoch
        for epoch in tqdm(range(epochs)):
            accTrain, losses = self.train_each_epoch(self.net, opti, train_loader, epoch, name)
            self.writer.add_scalar('training_accuracy/%s' % name, accTrain, epoch)

            acclist.append(accTrain)
            if lr_sched is not None:
                lr_sched.step(epoch)

            if (epoch + 1) % self.train_cfg.val_interval == 0:
                result = self.base_model_accuracy()
                print(result)
                current_acc = result['current']
                cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                str=cur_time + 'train epoch: %d  train acc %.4f cur acc %.4f loss: %.4f' % (epoch + 1, accTrain, current_acc, losses)
                print(str)
                self.logger.info(str)
                self.writer.add_scalar('test_accuracy/%s' % name, current_acc, epoch)
                if current_acc > best_acc:
                    best_acc = current_acc
        return best_acc

    def train_each_epoch(self, net, optimizer, dataloader, epoch, name):
        net.train()
        accFinal, tot_loss, iters = 0., 0., 0
        true_labels = []
        scaler = GradScaler()
        for Xs, ys, ind, _ in dataloader:
            Xs = Xs.cuda()
            ys = ys.cuda()
            true_labels.append(ys)
            # if self.is_amp == True:
            #     loss, y_hats = self.amp_train(net, scaler, optimizer, Xs, ys)
            # else:
            loss, y_hats = self.no_amp_train(net, optimizer, Xs, ys)

            tot_loss += loss.item()
            iters += 1
            accFinal += torch.sum((torch.max(y_hats, 1)[1] == ys).float()).data.item()
        self.writer.add_scalar('training_loss/%s' % name, tot_loss / (iters+1), epoch)
        # true_labels = torch.cat(true_labels, dim=0).cpu()
        # for i in range(self.train_params['n_class']):
        #     self.n_train_samples_each_class[i] += torch.sum(true_labels == i)
        return accFinal / len(dataloader.dataset), tot_loss / (iters+1)

    def amp_train(self, net, scaler, optimizer, Xs, ys):
        optimizer.zero_grad()
        with autocast():
            y_hats, feats = net(Xs)
            loss = F.cross_entropy(y_hats, ys)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss, y_hats

    def no_amp_train(self, net, optimizer, Xs, ys):
        optimizer.zero_grad()
        with torch.enable_grad():
            y_hats, feats = net(Xs)
            loss = F.cross_entropy(y_hats, ys)
        loss.backward()
        optimizer.step()
        return loss, y_hats

    def base_model_accuracy(self):
        self.net.eval()
        correct_num = 0
        ds_inds = []
        pred_cls = []
        true_labels = []

        loader = self.build_test_loader()
        for Xs, ys, ind, ds_ind in tqdm(loader):
            Xs = Xs.cuda()
            ys = ys.cuda()
            ds_inds.append(ds_ind)

            with torch.set_grad_enabled(False):
                y_hats, feats = self.net(Xs)
                _, preds = torch.max(y_hats, 1)
            true_labels.append(ys)

            correct_num += torch.sum(preds == ys.data)
            pred_cls.append(preds)

        pred_cls = torch.cat(pred_cls,dim=0)
        ds_inds = torch.cat(ds_inds, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        correct_mask = pred_cls==true_labels
        result={}
        # print(torch.sum(correct_mask), correct_num)
        # print(self.train_ds.domain_lens)
        # print(len(loader.dataset), self.train_ds.domain_lens[0]+self.train_ds.domain_lens[1]+self.train_ds.domain_lens[2]+self.train_ds.domain_lens[3])
        result['total']=(float(correct_num) / float(len(loader.dataset)))
        for i in range(len(self.test_ds.domain_lens)):
            key = 'ds'+str(i)
            value = torch.sum(correct_mask[ds_inds==i]) / float(self.test_ds.domain_lens[i])
            result[key]=value
        mask = ds_inds < self.train_cfg.dataset_number
        result['current'] = torch.sum(correct_mask[mask]) / float(torch.sum(mask))
        return result

    def test_per_category(self):
        self.net.eval()
        correct_num = 0
        ds_inds = []
        pred_cls = []
        true_labels = []

        loader = self.build_alldata_loader()
        for Xs, ys, ind, ds_ind in tqdm(loader):
            Xs = Xs.cuda()
            ys = ys.cuda()
            ds_inds.append(ds_ind)

            with torch.set_grad_enabled(False):
                y_hats, feats = self.net(Xs)
                _, preds = torch.max(y_hats, 1)
            true_labels.append(ys)

            correct_num += torch.sum(preds == ys.data)
            pred_cls.append(preds)

        pred_cls = torch.cat(pred_cls, dim=0)
        ds_inds = torch.cat(ds_inds, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        correct_mask = pred_cls == true_labels
        result = {}
        # print(torch.sum(correct_mask), correct_num)
        # print(self.train_ds.domain_lens)
        # print(len(loader.dataset), self.train_ds.domain_lens[0]+self.train_ds.domain_lens[1]+self.train_ds.domain_lens[2]+self.train_ds.domain_lens[3])
        result['total'] = (float(correct_num) / float(len(loader.dataset)))
        for i in range(len(self.test_ds.domain_lens)):
            key = 'ds' + str(i)
            value = torch.sum(correct_mask[ds_inds == i]) / float(self.test_ds.domain_lens[i])
            result[key] = value

            correct_num_per_cls = torch.zeros(self.net.n_label)
            total_num_per_cls = torch.zeros(self.net.n_label)
            ds_mask = ds_inds == i


            if i == 0:
                for c in range(self.net.n_label):
                    ds_true_labels = true_labels[ds_mask]
                    ds_pred_labels = pred_cls[ds_mask]
                    ds_correct = ds_true_labels == ds_pred_labels
                    total_num_per_cls[c] = torch.sum(ds_true_labels==c)
                    correct_num_per_cls[c] = torch.sum(ds_correct[ds_true_labels==c])
                print(correct_num_per_cls / total_num_per_cls)
                print(total_num_per_cls)
                return correct_num_per_cls / total_num_per_cls



    def test_feats(self):
        self.net.eval()
        ds_inds = []

        logits = []
        embedding_features = []
        true_labels = []
        total_preds = []

        loader = self.build_alldata_loader()
        for Xs, ys, ind, ds_ind in tqdm(loader):
            Xs = Xs.cuda()
            ys = ys.cuda()
            ds_inds.append(ds_ind)

            with torch.set_grad_enabled(False):
                y_hats, feats = self.net(Xs)
                _, preds = torch.max(y_hats, 1)

            embedding_features.append(feats)
            logits.append(y_hats)
            true_labels.append(ys)
            total_preds.append(preds)


        ds_inds = torch.cat(ds_inds, dim=0).cpu()
        logits = torch.cat(logits, dim=0).cpu()
        true_labels = torch.cat(true_labels, dim=0).cpu()
        total_preds = torch.cat(total_preds, dim=0).cpu()
        embedding_features = torch.cat(embedding_features, dim=0).cpu()

        return embedding_features, true_labels, total_preds, ds_inds, logits

