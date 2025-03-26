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
from torch.utils.data.sampler import RandomSampler, BatchSampler
import math
import timm
import os

class MarginTMSALTrainer(BaseTrainer):

    def __init__(self, net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg=None, is_amp=False, logger=None):
        super().__init__(net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg, is_amp, logger)
        self.multi_classifier = self.net[1]
        self.net = self.net[0]


    def train(self, name, n_epoch=None):
        self.center_loss = MultiCenterLoss(self.net.n_label, self.net.get_embedding_dim(),
                                           self.train_cfg.dataset_number)

        self.n_multi_classfiers = len(self.multi_classifier.multi_classifiers)

        opti, lr_sched = self.build_optimizer(self.net)
        multi_opti, multi_sched = self.build_multi_optimizer(self.multi_classifier)
        center_opti = torch.optim.Adadelta(self.center_loss.parameters(), lr=self.train_cfg.center_lr)

        # multi_opti, multi_sched = self.build_optimizer(self.multi_classifier)
        # center_opti, center_sched = self.build_optimizer(self.center_loss)

        train_loader = self.build_train_label_loader()
        acclist = []
        best_acc = 0.

        if n_epoch == None:
            epochs = self.train_cfg.epochs
        else:
            epochs = n_epoch

        best_backbone_params = copy.deepcopy(self.net)
        for epoch in tqdm(range(self.train_cfg.epochs)):
            accTrain, lossTrain = self.combine_train_each_epoch([opti, center_opti], train_loader, epoch, name)

            self.writer.add_scalar('training_accuracy/%s' % name, accTrain, epoch)
            acclist.append(accTrain)
            # lr_sched.step(epoch)
            if lr_sched is not None:
                lr_sched.step(epoch)
            # print('train loss %.4f, acc train %.4f'%(lossTrain, accTrain))
            if (epoch + 1) % self.train_cfg.val_interval == 0:
                result = self.base_model_accuracy()
                print(result)
                self.logger.info(result)
                current_acc = result['current']

                if current_acc > best_acc:
                    print('save net ')
                    best_acc = current_acc
                    best_backbone_params = copy.deepcopy(self.net.state_dict())
                cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                str = cur_time + 'train epoch: %d  train acc %.4f cur acc %.4f loss %.4f' % (
                epoch + 1, accTrain, current_acc, lossTrain)
                print(str)
                self.logger.info(str)
                self.writer.add_scalar('test_accuracy/%s' % name, current_acc, epoch)
                # if current_acc > best_acc:
                #     best_acc = current_acc

            if epoch + 1 == self.train_cfg.center_epoch:
                centers = self.calculate_center()
                self.center_loss.initial_centers(centers)
        self.net.load_state_dict(best_backbone_params)
        # del best_net_params

        self.freeze(self.net)
        best_acc = 0.

        if self.train_cfg.is_multi_classifier == True:
            for i, params in enumerate(self.multi_classifier.multi_classifiers):
                self.copy_params(self.multi_classifier.multi_classifiers[i], self.net.fc)

            best_multi_acc_list = []
            # best_multi_params = copy.deepcopy(self.multi_classifier.multi_classifiers)
            # best_backbone_params = copy.deepcopy(self.net)

            for i in range(self.n_multi_classfiers):
                best_multi_acc_list.append(0.)
            # center_sim_thrs = self.get_center_sim_thr()
            # self.center_sim_thrs = center_sim_thrs
            # sim_thrs, l_norm_feats = self.get_sim_thr()
            for epoch in tqdm(range(self.train_cfg.multi_epochs)):
                # ori_num, cur_num = self.multi_classifier_train_each_epoch(multi_opti, train_loader, epoch, sim_thrs, l_norm_feats)
                ori_num, cur_num = self.multi_classifier_center_sim_train_each_epoch(multi_opti, train_loader, epoch)
                # print(ori_num, cur_num)
                # multi_sched.step(epoch)
                if multi_sched is not None:
                    multi_sched.step(epoch)
                if (epoch + 1) % 10 == 0:
                    result, acclist = self.multi_classifier_test()
                    print(result)
                    self.logger.info(result)
                    current_acc = result['current']
                    if current_acc > best_acc:
                        best_acc = current_acc
                # #         best_backbone_params = copy.deepcopy(self.net.state_dict())
                # #         best_multi_params = copy.deepcopy(self.multi_classifier.state_dict())
                # #
                # #     # for i in range(len(acclist)):
                # #     #     if acclist[i] > best_multi_acc_list[i]:
                # #     #         print('save multi ', i)
                # #     #         best_multi_acc_list[i] = acclist[i]
                # #     #         best_multi_params[i] = copy.deepcopy(self.multi_classifier.multi_classifiers[i])
                # #     # if current_acc > best_multi_acc:
                # #     #     best_multi_acc = current_acc
                # #     #     best_multi_params = copy.deepcopy(self.multi_classifier.state_dict())
                #     cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                #     str = cur_time + 'multi classifier cur acc %.4f' % (current_acc)
                #     print(str)
                #     self.logger.info(str)
                #     print(result)
                #     self.logger.info(result)
            print('ori num ', ori_num)
            print('cur num ', cur_num)
            self.logger.info(ori_num)
            self.logger.info(cur_num)


            # self.copy_params(self.net, best_backbone_params)
            # self.copy_params(self.multi_classifier.multi_classifiers, best_multi_params)

            # self.net.load_state_dict(best_backbone_params)
            # self.multi_classifier.load_state_dict(best_multi_params)
            # for i, params in enumerate(best_multi_params):
            #     self.copy_params(self.multi_classifier.multi_classifiers[i], params)
            # self.multi_classifier.multi_classifiers.load_state_dict(best_multi_params)
            # del best_multi_params
            # del best_backbone_params

            # result, acclist = self.multi_classifier_test()
            # print(result)

        return best_acc

    def freeze(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def build_multi_optimizer(self, net):
        # create optimizer
        optimizer_cfg = self.train_cfg.multi_optimizer

        if optimizer_cfg.type == 'AdamW':
            optimizer = optim.AdamW(net.parameters(), lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay,
                                    eps=optimizer_cfg.eps, betas=optimizer_cfg.betas)
        elif optimizer_cfg.type == 'Adadelta':
            optimizer = optim.Adadelta(net.parameters(), lr=optimizer_cfg.lr)
        else:
            raise 'incorrect optimizer'

        scheduler_cfg = self.train_cfg.multi_scheduler
        lr_sched = None
        if scheduler_cfg is not None:
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

        # return optimizer, lr_sched

    def multi_classifier_test(self):
        self.net.eval()
        self.multi_classifier.eval()
        correct_num = 0
        ds_inds = []
        pred_cls = []
        true_labels = []


        loader = self.build_test_loader()
        result = {}

        for Xs, ys, ind, ds_ind in tqdm(loader):
            Xs = Xs.cuda()
            ys = ys.cuda()

            with torch.set_grad_enabled(False):

                # y_hats, feats = self.net(Xs)
                for i in range(self.train_cfg.dataset_number):
                    mask = ds_ind == i
                    # if torch.sum(mask == 0):
                    #     continue
                    ds_Xs = Xs[ds_ind == i]
                    if len(ds_Xs) == 0:
                        continue
                    _, ds_feats = self.net(ds_Xs)
                    ds_feats = ds_feats.detach()
                    ds_ys = ys[ds_ind == i]
                    assert len(ds_Xs) == len(ds_ys)
                    # ds_feats = F.normalize(ds_feats, dim=1)
                    ds_y_hats = self.multi_classifier(ds_feats, i)
                    _, ds_preds = torch.max(ds_y_hats, 1)

                    true_labels.append(ds_ys)
                    pred_cls.append(ds_preds)
                    ds_inds.append(torch.tensor([i] * len(ds_ys)))

        pred_cls = torch.cat(pred_cls, dim=0)
        ds_inds = torch.cat(ds_inds, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        correct_mask = pred_cls == true_labels
        result['total'] = (float(correct_num) / float(len(loader.dataset)))
        acc_list = []
        for i in range(self.train_cfg.dataset_number):
            key = 'ds' + str(i)
            value = torch.sum(correct_mask[ds_inds == i]) / float(self.test_ds.domain_lens[i])
            acc_list.append(value)
            result[key] = value
        mask = ds_inds < self.train_cfg.dataset_number
        result['current'] = torch.sum(correct_mask[mask]) / float(torch.sum(mask))
        result['total'] = result['current']
        return result, acc_list

    def multi_classifier_center_sim_train_each_epoch(self, optimizer, train_loader, epoch):
        self.net.train()
        self.multi_classifier.train()
        accFinal, tot_loss, iters = 0., 0., 0

        total_correct = 0
        ori_num = torch.zeros(self.train_cfg.dataset_number)
        cur_num = torch.zeros(self.train_cfg.dataset_number)
        for Xs, ys, ind, ds_ind in train_loader:
            Xs = Xs.cuda()
            ys = ys.cuda()
            ds_ind = ds_ind.cuda()
            with torch.no_grad():
                preds, feats = self.net(Xs)
                feats = feats.detach()

            classifier_loss = 0.
            with torch.enable_grad():
                for i in range(self.train_cfg.dataset_number):
                    if self.train_cfg.is_recall == True:
                        # ds_train_mask = self.cal_center_sim_train_inds(feats, ds_ind, i, ys, sim_thrs)
                        # ds_train_mask = self.cal_margin_train_inds(feats, ds_ind, i, ys, preds)
                        ds_train_mask = self.cal_margin_grad(feats, ds_ind, i, ys, preds)
                        cur_num[i] = cur_num[i] + torch.sum(ds_train_mask)
                        ori_num[i] = ori_num[i] + torch.sum(ds_ind == i)
                    else:
                        ds_train_mask = ds_ind == i
                    ds_feats = feats[ds_train_mask]
                    # ds_feats = F.normalize(ds_feats, dim=1)
                    ds_ys = ys[ds_train_mask]
                    y_hats = self.multi_classifier(ds_feats, i)
                    ds_classifier_loss = F.cross_entropy(y_hats, ds_ys)
                    classifier_loss += ds_classifier_loss
                    # if self.train_cfg.is_margin==True:
                    #     ds_margin_loss = Margin_loss(F.normalize(y_hats, dim=1), ds_ys)
                    #     classifier_loss+=ds_margin_loss

                    correct = torch.sum(torch.max(y_hats, 1)[1] == ds_ys)
                    total_correct += correct
            # loss_multi = self.multi_classifier
            total_loss = (classifier_loss) / self.train_cfg.dataset_number
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            tot_loss += total_loss
        iters += iters + 1

        return ori_num, cur_num


    def cal_center_sim_train_inds(self, feats, ds_ind, target_ds_ind, ys, sim_thrs):
        target_sim_thrs = sim_thrs[target_ds_ind]
        ds_mask = ds_ind == target_ds_ind
        tmp_mask = ds_ind != target_ds_ind
        ds_train_mask = torch.zeros(len(feats), dtype=torch.bool).cuda() | ds_mask
        norm_feats = F.normalize(feats.detach(), dim=1)[tmp_mask]
        target_ds_centers = F.normalize(self.center_loss.centers[target_ds_ind], dim=1)
        sims = torch.mm(norm_feats, target_ds_centers.T)
        target_sim_thrs = target_sim_thrs.repeat((len(sims), 1)).cuda()
        mask = sims > target_sim_thrs
        mask = torch.sum(mask, dim=1) > 0.
        ds_train_mask = torch.masked_scatter(ds_train_mask, tmp_mask, mask)
        return ds_train_mask

    def cal_margin_grad(self, feats, ds_ind, target_ds_ind, ys, preds):
        ds_classifier = self.multi_classifier.multi_classifiers[target_ds_ind]

        _, preds_cls = torch.max(preds, 1)
        pred_true_mask = preds_cls == ys
        ds_mask = ds_ind == target_ds_ind
        ds_train_mask = torch.zeros(len(feats), dtype=torch.bool).cuda() | ds_mask

        ds_feats = feats[ds_mask & pred_true_mask]
        ds_o_feats = feats[~ds_mask]
        ds_ys = ys[ds_mask & pred_true_mask]
        ds_o_ys = ys[~ds_mask]
        ds_ys_inds = torch.unsqueeze(ds_ys, dim=1)

        ds_logits = torch.mm(ds_feats, ds_classifier.weight.T)
        ds_ys_logits = torch.gather(input=ds_logits, dim=1, index=ds_ys_inds).squeeze()
        ds_ys_logits = ds_ys_logits.repeat((self.net.n_label, 1)).T

        ds_delta_logits = ds_ys_logits - ds_logits
        # print(ds_delta_logits)


        # print(ds_ys_inds.shape)
        # exit()

        ds_o_mask = torch.zeros(len(ds_o_feats), dtype=torch.bool).cuda()
        for i in range(len(ds_o_feats)):
            ds_classifier.zero_grad()
            single_feat = torch.unsqueeze(ds_o_feats[i], dim=0).requires_grad_()
            single_hats = ds_classifier(single_feat)
            single_label = torch.full([1], ds_o_ys[i]).cuda()
            # print(single_label)
            loss_ce = F.cross_entropy(single_hats, single_label)
            # loss_margin = Margin_loss(single_hats, single_label)
            loss = loss_ce# + loss_margin
            single_grad = torch.autograd.grad(outputs=loss, inputs=ds_classifier.weight, retain_graph=True)[0]

            mask = ds_ys == ds_o_ys[i]

            ds_grad = torch.mm(ds_feats, single_grad.T)
            ds_ys_grad = torch.gather(input=ds_grad, dim=1, index=ds_ys_inds).squeeze()
            ds_ys_grad = ds_ys_grad.repeat((self.net.n_label, 1)).T
            delta = ds_ys_grad - ds_grad
            mask = self.train_cfg.multi_lr * delta <= ds_delta_logits #/ self.train_cfg.multi_epochs
            sum_count = torch.sum(~mask)
            if sum_count == 0:
                ds_o_mask[i] = True
            #
            mask = ds_ys == ds_o_ys[i]

            out = torch.mm(ds_feats[mask], single_feat.T)
            # print(ds_feats[mask].shape, single_feat.shape)
            if torch.sum(out < 0.) > 0:
                ds_o_mask[i] = False

            # print(ds_delta_logits)
            # print(delta_grad)
            # exit()
        ds_train_mask = torch.masked_scatter(ds_train_mask, ~ds_mask, ds_o_mask)
        return ds_train_mask


    def cal_margin_train_inds(self, feats, ds_ind, target_ds_ind, ys, preds):
        ds_classifier = self.multi_classifier.multi_classifiers[target_ds_ind]

        _, preds_cls = torch.max(preds, 1)
        pred_true_mask = preds_cls == ys
        ds_mask = ds_ind == target_ds_ind
        ds_train_mask = torch.zeros(len(feats), dtype=torch.bool).cuda() | ds_mask

        ds_feats = feats[ds_mask&pred_true_mask]
        ds_o_feats = feats[~ds_mask]
        ds_ys = ys[ds_mask&pred_true_mask]
        ds_o_ys = ys[~ds_mask]
        ds_ys_inds = (torch.unsqueeze(ds_ys, dim=1))


        ds_logits = torch.mm(ds_feats, ds_classifier.weight.T)
        ds_ys_logits = torch.gather(input=ds_logits, dim=1, index=ds_ys_inds).squeeze()
        ds_ys_logits = ds_ys_logits.repeat((self.net.n_label, 1)).T
        for i in range(len(ds_feats)):
            ds_ys_logits[i][ds_ys[i]] = 999999.0
        ds_delta_logits = ds_ys_logits - ds_logits

        ds_o_mask = torch.zeros(len(ds_o_feats), dtype=torch.bool).cuda()
        for i in range(len(ds_o_feats)):
            ds_classifier.zero_grad()
            single_feat = torch.unsqueeze(ds_o_feats[i], dim=0).requires_grad_()
            single_hats = ds_classifier(single_feat)
            single_label = torch.full([1], ys[~ds_mask][i]).cuda()
            loss_ce = F.cross_entropy(single_hats, single_label)
            loss_margin = Margin_loss(single_hats, single_label)
            loss = loss_ce + loss_margin
            loss.backward(retain_graph=True)
            # print(loss)

            single_grad = ds_classifier.weight.grad.detach()
            delta_grad = torch.mm(ds_feats, single_grad.T)
            mask = ds_delta_logits > delta_grad * self.train_cfg.multi_lr
            sum_count = torch.sum(~mask)
            if sum_count == 0:
                ds_o_mask[i] = True

            mask = ds_ys == ds_o_ys[i]
            out = torch.mm(ds_feats[mask], single_feat.T)
            # print(ds_feats[mask].shape, single_feat.shape)
            if torch.sum(out < 0.) > 0:
                ds_o_mask[i] = False

            # print(ds_delta_logits)
            # print(delta_grad)
        ds_train_mask = torch.masked_scatter(ds_train_mask, ~ds_mask, ds_o_mask)
        return ds_train_mask


    def cal_grad_train_inds(self, feats, ds_ind, target_ds_ind, ys, preds):
        # print(target_ds_ind, ds_ind)
        ds_mask = ds_ind == target_ds_ind
        ds_f_mask = ds_ind != target_ds_ind
        ds_train_mask = torch.zeros(len(feats), dtype=torch.bool).cuda() | ds_mask
        ds_classifier = self.multi_classifier.multi_classifiers[target_ds_ind]

        ds_centers = self.center_loss.centers[target_ds_ind]
        ds_norm_grad = torch.zeros((self.net.n_label, feats.shape[1])).cuda()
        for i in range(len(ds_centers)):
            ds_centers_single_feat = torch.unsqueeze(ds_centers[i], dim=0).requires_grad_()
            ds_centers_single_labels = torch.full([1], i).cuda()
            ds_centers_single_hats = ds_classifier(ds_centers_single_feat)

            ds_classifier.zero_grad()
            loss_ce = F.cross_entropy(ds_centers_single_hats, ds_centers_single_labels)
            loss_margin = Margin_loss(ds_centers_single_hats, ds_centers_single_labels)
            loss = loss_ce# + loss_margin
        # print(loss)
        # print(ds_center_hats)

            loss.backward(retain_graph=True)
            ds_single_grad = ds_classifier.weight.grad.detach()
            # ds_grad = torch.autograd.grad(outputs=loss, inputs=ds_centers, retain_graph=True)[0]
            ds_single_norm_grad = F.normalize(ds_single_grad, dim=1)
            print(ds_single_norm_grad)
            ds_norm_grad[i] = ds_single_norm_grad[i]
        print(ds_norm_grad)
        exit()

        for i in range(len(feats)):
            ds_classifier.zero_grad()
            single_feat = torch.unsqueeze(feats[i], dim=0).requires_grad_()
            single_hats = ds_classifier(single_feat)
            single_label = torch.full([1], ys[i]).cuda()
            loss_ce = F.cross_entropy(single_hats, single_label)
            loss_margin = Margin_loss(single_hats, single_label)
            loss = loss_ce + loss_margin
            loss.backward(retain_graph=True)
            print(loss)

            # single_grad = torch.autograd.grad(outputs=loss, inputs=single_feat, retain_graph=True)[0]
            single_grad = ds_classifier.weight.grad.detach()
            # print(single_grad)
            single_norm_grad = F.normalize(single_grad, dim=1)
            print(single_norm_grad.shape)
            sim = (single_norm_grad * ds_norm_grad).sum(dim=-1)
            # sim = (torch.mm(single_norm_grad,  ds_norm_grad.T))
            print(sim.shape)
            sim_max = torch.argmin(sim)

            print(ds_ind[i]==target_ds_ind, sim_max==ys[i] , sim_max, ys[i], sim[ys[i]])
            # print('single', single_grad.shape, sim.shape)
            # print(sim)
        #     print(label)
        #     print(y_hats.shape, single_feat.shape, ds_centers.shape)
        #     loss_ce = F.cross_entropy(y_hats, label)
        #     # ds_margin_loss =
        #     loss_margin = Margin_loss(y_hats, label)
        #     # loss_ce = loss_ce + lo
        #     loss = loss_ce + loss_margin
        #     loss.backward(retain_graph=True)
        #     print(ds_classifier.weight.grad.shape)
        #     delta = torch.mm(ds_classifier.weight, ds_centers.T).T
        #     # grad_delta =
        #     diag = torch.diag(delta)
        #     print(diag.shape)
        #     diag = diag.repeat((ds_centers.shape[0], 1)).T
        #     delta = diag - delta
        #     print(delta)
        #     # torch.diagonal()
        #     # print(diag.shape)
        #     # print(diag)
        #     tmp = torch.mm(ds_classifier.weight.grad, ds_centers.T).T
        #     # tmp = torch.dia
        #     print(delta >= tmp)
        #     print(delta)
        #     print(tmp)


        # exit()
        # for i in range(self.n_multi_classfiers):
        #     ds_centers = F.normalize(self.center_loss.centers[i], dim=1)
        ds_classifier.zero_grad()


    def combine_train_each_epoch(self, optimizer, train_loader, epoch, name):
        self.net.train()
        accFinal, tot_loss, iters = 0., 0., 0

        net_opit = optimizer[0]
        center_opit = optimizer[1]

        ct_weight = self.get_center_weight(epoch)
        # ct_weight = 1.0

        for Xs, ys, ind, ds_ind in train_loader:
            Xs = Xs.cuda()
            ys = ys.cuda()
            ds_ind = ds_ind.cuda()
            # print(ds_ind)
            with torch.enable_grad():
                y_hats, feats = self.net(Xs)
                if self.train_cfg.is_center==False:
                    ct_weight = 0.
                loss_ct = self.center_loss(feats, ys, ds_ind, self.logger)  * ct_weight
                loss_ce = F.cross_entropy(y_hats, ys)
                if self.train_cfg.is_margin==True:
                    loss_margin = Margin_loss(F.normalize(y_hats, dim=1), ys, margin=self.train_cfg.margin)
                    total_losses = loss_ce + loss_ct + loss_margin
                    # print(loss_ce, loss_ct, loss_margin)
                else:
                    total_losses = loss_ce + loss_ct
            accFinal += torch.sum((torch.max(y_hats, 1)[1] == ys).float()).data.item()

            net_opit.zero_grad()
            center_opit.zero_grad()
            total_losses.backward()
            net_opit.step()
            if self.train_cfg.is_center == True:
                for param in self.center_loss.parameters():
                    param.grad.data *= (1./(ct_weight + 1e-12))
                center_opit.step()

        iters += 1
        tot_loss += total_losses
        # print('epoch %d training loss %.4f %.4f'%(epoch, loss_ct, loss_ce))
        self.writer.add_scalar('training_loss/%s' % name, tot_loss / iters, epoch)

        return accFinal / len(self.label_info.label_ind), tot_loss / iters

    def get_center_weight(self, epoch):
        center_weight = self.train_cfg['center_weights'][0]
        for i, ms in enumerate(self.train_cfg['center_milestones']):
            if epoch >= ms:
                center_weight = self.train_cfg['center_weights'][i]
        # self.logger.info('Center Weight: {}'.format(center_weight))
        return center_weight

    def predict_embed(self, data_loader, eval=True):
        embedding_features = []
        true_labels = []
        ds_inds = []

        if eval:
            self.net.eval()
        else:
            self.net.train()

        for x, y, ind, ds_ind in data_loader:
            x, y, ds_ind = x.cuda(), y.cuda(), ds_ind.cuda()
            with torch.no_grad():
                out, e1 = self.net(x)

            true_labels.append(y)
            embedding_features.append(e1)
            ds_inds.append(ds_ind)

        true_labels = torch.cat(true_labels, dim=0)
        embedding_features = torch.cat(embedding_features, dim=0)
        ds_inds = torch.cat(ds_inds, dim=0)

        return embedding_features, true_labels, ds_inds


    def pred_margin_embed(self, data_loader, eval=True):
        embedding_features = []
        true_labels = []
        ds_inds = []

        if eval:
            self.net.eval()
        else:
            self.net.train()

        for x, y, ind, ds_ind in data_loader:
            x, y, ds_ind = x.cuda(), y.cuda(), ds_ind.cuda()
            out, feats = self.net(x)
            with torch.no_grad():

                for i in range(self.train_cfg.dataset_number):
                    ds_train_mask = ds_ind == i
                    ds_feats = feats[ds_train_mask]
                    # ds_feats = F.normalize(ds_feats, dim=1)
                    # ds_ys = ys[ds_train_mask]
                    ds_ys = y[ds_train_mask]
                    ds_y_hats = self.multi_classifier(ds_feats, i)
                    _, preds = torch.max(ds_y_hats, 1)

                    mask = preds == ds_ys
                    true_labels.append(ds_ys[mask])
                    embedding_features.append(ds_feats[mask])
                    ds_inds.append(ds_ind[ds_train_mask][mask])

        true_labels = torch.cat(true_labels, dim=0)
        embedding_features = torch.cat(embedding_features, dim=0)
        ds_inds = torch.cat(ds_inds, dim=0)

        return embedding_features, true_labels, ds_inds

    def cal_domain_sim_thr(self):
        '''same as the paper'''
        sim_thr = torch.ones((self.n_multi_classfiers, self.net.n_label)).cuda()
        print('start cal domain sim thr')
        train_loader = self.build_train_label_loader()
        for Xs, ys, ind, ds_ind in train_loader:
            Xs = Xs.cuda()
            ys = ys.cuda()
            ds_ind = ds_ind.cuda()
            with torch.no_grad():
                preds, feats = self.net(Xs)
                feats = feats.detach()
                for target_ds_ind in range(self.train_cfg.dataset_number):
                    # target_ds_ind = i
                    ds_centers = F.normalize(self.center_loss.centers[target_ds_ind], dim=1)
                    ds_classifier = self.multi_classifier.multi_classifiers[target_ds_ind]
                    _, preds_cls = torch.max(preds, 1)
                    pred_true_mask = preds_cls == ys
                    ds_mask = ds_ind == target_ds_ind

                    ds_feats = feats[ds_mask & pred_true_mask]
                    ds_o_feats = feats[~ds_mask]
                    ds_ys = ys[ds_mask & pred_true_mask]
                    ds_o_ys = ys[~ds_mask]

                    ds_ys_inds = torch.unsqueeze(ds_ys, dim=1)
                    ds_logits = torch.mm(ds_feats, ds_classifier.weight.T)
                    ds_ys_logits = torch.gather(input=ds_logits, dim=1, index=ds_ys_inds).squeeze()
                    ds_ys_logits = ds_ys_logits.repeat((self.net.n_label, 1)).T
                    ds_delta_logits = ds_ys_logits - ds_logits

                    ds_o_mask = torch.zeros(len(ds_o_feats), dtype=torch.bool).cuda()
                    # print(ds_o_feats.shape)
                    with torch.enable_grad():
                        for i in range(len(ds_o_feats)):
                            ds_classifier.zero_grad()
                            single_feat = torch.unsqueeze(ds_o_feats[i], dim=0).requires_grad_()
                            single_hats = ds_classifier(single_feat)
                            single_label = torch.full([1], ds_o_ys[i]).cuda()
                            loss_ce = F.cross_entropy(single_hats, single_label)
                            loss = loss_ce
                            single_grad = torch.autograd.grad(outputs=loss, inputs=ds_classifier.weight, retain_graph=True)[
                                0]
                            ds_grad = torch.mm(ds_feats, single_grad.T)
                            ds_ys_grad = torch.gather(input=ds_grad, dim=1, index=ds_ys_inds).squeeze()
                            ds_ys_grad = ds_ys_grad.repeat((self.net.n_label, 1)).T
                            delta = ds_ys_grad - ds_grad

                            mask = self.train_cfg.multi_lr * delta <= ds_delta_logits
                            sum_count = torch.sum(~mask)
                            if sum_count == 0:
                                ds_o_mask[i] = True
                            mask = ds_ys == ds_o_ys[i]

                            out = torch.mm(ds_feats[mask], single_feat.T)
                            if torch.sum(out < 0.) > 0:
                                ds_o_mask[i] = False
                    ds_o_domain_feats = ds_o_feats[ds_o_mask]
                    ds_o_domain_feats = F.normalize(ds_o_domain_feats, dim=1)
                    ds_o_domain_ys = ds_o_ys[ds_o_mask]
                    ds_o_domain_centers = ds_centers[ds_o_domain_ys]
                    cos_sim = (ds_o_domain_feats * ds_o_domain_centers).sum(dim=-1)
                    for ind in range(len(cos_sim)):
                        tmp_sim = cos_sim[ind]
                        tmp_label = ds_o_domain_ys[ind]
                        if sim_thr[target_ds_ind][tmp_label] > tmp_sim:
                            sim_thr[target_ds_ind][tmp_label] = tmp_sim
        for i in range(self.n_multi_classfiers):
            for j in range(self.net.n_label):
                if sim_thr[i][j] == 1.0:
                    sim_thr[i][j] = 0.0
        return sim_thr


    def get_center_sim_thr(self):
        sim_thr = torch.zeros((self.n_multi_classfiers, self.net.n_label))
        label_loader = self.build_ds_sequence_train_label_loader()
        # l_embedding, l_true_labels, l_ds_inds = self.predict_embed(label_loader)
        l_embedding, l_true_labels, l_ds_inds = self.pred_margin_embed(label_loader)
        l_norm_feats = F.normalize(l_embedding, dim=1)
        for i in range(self.n_multi_classfiers):
            ds_centers = F.normalize(self.center_loss.centers[i], dim=1)
            ds_mask = (l_ds_inds == i)
            for j in range(self.net.n_label):
                ds_c_centers = ds_centers[j]
                c_mask = (l_true_labels == j)
                ds_c_mask = ds_mask & c_mask
                feats = l_norm_feats[ds_c_mask]

                cos_sim = (feats * ds_c_centers).sum(dim=-1)
                if len(cos_sim) == 0:
                    sim_thr[i][j] = 0.
                else:
                    sim_thr[i][j] = torch.min(cos_sim)
                    # if self.train_cfg.alpha == 1.0:
                    #     sim_thr[i][j] = torch.min(cos_sim)
                    # else:
                    #     sim_thr[i][j] = torch.sort(cos_sim, descending=True).values[math.floor(len(cos_sim)*self.train_cfg.alpha)]
                    # sim_thr[i][j] = torch.mean(cos_sim)
        print(sim_thr)
        return sim_thr

                # ds_c_feats



    def get_sim_thr(self):
        label_loader = self.build_ds_sequence_train_label_loader()
        l_embedding, l_true_labels, l_ds_inds = self.predict_embed(label_loader)
        l_norm_feats = F.normalize(l_embedding, dim=1)
        sim = torch.mm(l_norm_feats, l_norm_feats.T) - torch.eye(len(l_norm_feats)).cuda()
        label_slices = []
        index = 0
        l_ds_norm_feats=[]
        for i in range(len(self.multi_classifier.multi_classifiers)):
            ds_count = torch.sum(l_ds_inds == i)
            label_slices.append((copy.deepcopy(index), index + ds_count))
            index += ds_count
            l_ds_norm_feats.append(l_norm_feats[l_ds_inds == i])
        sim_thrs = []
        for slice_tuple in label_slices:
            sim_block = sim[slice_tuple[0]:slice_tuple[1], slice_tuple[0]:slice_tuple[1]]
            max_sim = torch.max(sim_block, dim=1).values
            sim_thrs.append(max_sim)
        return sim_thrs, l_ds_norm_feats


    def get_multi_classifier_mask(self):
        label_loader = self.build_train_label_loader(shuffle=False)
        l_embedding, l_true_labels, l_ds_inds = self.predict_embed(label_loader)
        l_norm_feats = F.normalize(l_embedding, dim=1)
        sim = torch.mm(l_norm_feats, l_norm_feats.T) - torch.eye(len(l_norm_feats)).cuda()
        label_slices = []
        index = 0
        for i in range(len(self.multi_classifier.multi_classifiers)):
            ds_count = torch.sum(l_ds_inds==i)
            label_slices.append((copy.deepcopy(index), index+ds_count))
            index += ds_count

        sim_thr = []
        for slice_tuple in label_slices:
            sim_block = sim[slice_tuple[0]:slice_tuple[1], slice_tuple[0]:slice_tuple[1]]
            max_sim = torch.max(sim_block, dim=1).values
            sim_thr.append(max_sim)

        multi_masks = []
        for i in range(len(self.multi_classifier.multi_classifiers)):
            ds_mask = (l_ds_inds == i)
            ds_train_mask = torch.zeros(len(l_embedding), dtype=torch.bool).cuda() | ds_mask
            tmp_mask = l_ds_inds != i
            tmp_sim = torch.mm(l_norm_feats[tmp_mask], l_norm_feats[l_ds_inds == i].T)
            sorted_result = torch.max(tmp_sim, dim=1)
            sorted_ind = sorted_result.indices
            sorted_values = sorted_result.values
            ds_sim_thr = sim_thr[i]
            ds_sim_thr = ds_sim_thr[sorted_ind]
            mask = sorted_values > ds_sim_thr
            # print(torch.sum(mask), mask.shape)
            # print('before:', torch.sum(ds_train_mask))
            ds_train_mask = torch.masked_scatter(ds_train_mask, tmp_mask, mask)
            # print('after:', torch.sum(ds_train_mask))
            multi_masks.append(ds_train_mask)
            str = 'classifier %d train count %d to %d' % (i, torch.sum(ds_mask), torch.sum(ds_train_mask))
            print(str)
            self.logger.info(str)
        return multi_masks, l_embedding, l_true_labels


    def calculate_center(self):
        centers = torch.zeros_like(self.center_loss.centers)
        train_loader = self.build_train_label_loader()
        logits = []
        embedding_features = []
        ds_inds = []
        true_labels = []
        for x, y, ind, ds_ind in train_loader:
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                out, e1 = self.net(x)

            logits.append(out)
            true_labels.append(y)

            embedding_features.append(e1)
            ds_inds.append(ds_ind)
            # probs.append(prob)
        embedding_features = torch.cat(embedding_features, dim=0).cpu()
        true_labels = torch.cat(true_labels, dim=0).cpu()
        ds_inds = torch.cat(ds_inds, dim=0).cpu()
        for ds in range(self.train_cfg.dataset_number):
            for label in range(self.net.n_label):
                mask = (true_labels == label) & (ds_inds == ds)
                if torch.sum(mask) == 0:
                    centers[ds][label] = centers[0][label]
                # print(torch.sum(mask))
                else:
                    centers[ds][label] = torch.mean(embedding_features[mask], dim=0)
        # print('centers nan:', torch.sum(torch.isnan(centers)))

        return centers.cuda()

    def copy_params(self, target, source):
        for param_t, param_s in zip(target.parameters(),
                                    source.parameters()):
            param_t.data = param_s.data

