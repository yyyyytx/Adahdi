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
class DiscriminatorMultiClassifierTrainer(BaseTrainer):

    def __init__(self, net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg=None, is_amp=False, logger=None):
        super().__init__(net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg, is_amp, logger)
        self.net = net[0]
        self.domain_discriminator = net[1]
        self.multi_classifier = net[2]
        self.multi_discriminator = net[3]


    def train(self, name, n_epoch=None):

        opti, lr_sched = self.build_optimizer(self.net)
        domain_opti, domain_sched = self.build_optimizer(self.domain_discriminator)
        multi_classifier_opti = torch.optim.Adam(self.multi_classifier.parameters_list(), lr=0.01)
        multi_discriminator_opti = torch.optim.AdamW(self.multi_discriminator.parameters_list(), lr=0.1)

        train_loader = self.build_train_label_loader()
        acclist = []
        best_acc = 0.

        if n_epoch == None:
            epochs = self.train_cfg.epochs
        else:
            epochs = n_epoch

        for epoch in tqdm(range(self.train_cfg.epochs)):
            accTrain, lossTrain = self.combine_train_each_epoch([opti, domain_opti], train_loader, epoch, name)

            self.writer.add_scalar('training_accuracy/%s' % name, accTrain, epoch)
            acclist.append(accTrain)
            lr_sched.step()
            domain_sched.step()

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

        # multi_classifier_mask, l_norm_feats, l_true_labels = self.get_multi_classifier_mask()
        # self.multi_classifier_train_each_epoch(multi_opti, multi_classifier_mask, l_norm_feats, l_true_labels)
        sim_thrs, l_norm_feats = self.get_sim_thr()
        for epoch in tqdm(range(self.train_cfg.multi_epochs)):
            accTrain, lossTrain = self.multi_classifier_train_each_epoch(multi_opti, train_loader, epoch, sim_thrs, l_norm_feats)

            if (epoch + 1) % 10 == 0:
                result = self.multi_classifier_test()
                current_acc = result['current']
                cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                # str = cur_time + 'multi classifier cur acc %.4f' % (current_acc)
                # print(str)
                # self.logger.info(str)
                print(result)
                self.logger.info(result)


        return best_acc


    def build_multi_optimizer(self, net):
        # create optimizer
        optimizer_cfg = self.train_cfg.optimizer
        if optimizer_cfg.type == 'SGD':
            optimizer = optim.SGD(net.parameters_list(), lr=optimizer_cfg.lr,
                                  momentum=optimizer_cfg.momentum,
                                  weight_decay=optimizer_cfg.weight_decay)
        # elif self.train_params['opt'] == 'ADAMW':
        #     optimizer = optim.AdamW(self.net.parameters(), lr=self.train_params['lr'])
        else:
            raise 'incorrect optimizer'

        scheduler_cfg = self.train_cfg.scheduler
        if scheduler_cfg.type == 'CosineAnnealingLR':
            lr_sched = CosineAnnealingLR(optimizer, self.train_cfg.epochs)
        elif scheduler_cfg.type == 'MultiStepLR':
            lr_sched = MultiStepLR(optimizer, milestones=scheduler_cfg.milestones)
        else:
            raise 'incorrect lr sched'

        return optimizer, lr_sched


    def multi_classifier_test(self):
        self.net.eval()
        correct_num = 0
        ds_inds = []
        pred_cls = []
        true_labels = []

        loader = self.build_test_loader()
        result = {}

        for Xs, ys, ind, ds_ind in loader:
            Xs = Xs.cuda()
            ys = ys.cuda()

            with torch.set_grad_enabled(False):
                y_hats, feats = self.net(Xs)
                for i in range(self.train_cfg.dataset_number):
                    ds_feats = feats[ds_ind == i].detach()
                    ds_ys = ys[ds_ind == i]
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
        for i in range(self.train_cfg.dataset_number):
            key = 'ds' + str(i)
            value = torch.sum(correct_mask[ds_inds == i]) / float(self.test_ds.domain_lens[i])
            result[key] = value
        mask = ds_inds < self.train_cfg.dataset_number
        result['current'] = torch.sum(correct_mask[mask]) / float(torch.sum(mask))
        return result

    def multi_classifier_train_each_epoch(self, optimizer, train_loader, epoch, sim_thrs, l_norm_feats):
        self.net.train()
        self.multi_classifier.train()
        accFinal, tot_loss, iters = 0., 0., 0

        total_correct = 0
        for Xs, ys, ind, ds_ind in train_loader:
            Xs = Xs.cuda()
            ys = ys.cuda()
            ds_ind = ds_ind.cuda()
            with torch.no_grad():
                _, feats = self.net(Xs)

            classifier_loss = 0.
            with torch.enable_grad():
                for i in range(self.train_cfg.dataset_number):
                    ds_train_mask = self.cal_train_inds(feats, ds_ind, i, sim_thrs, l_norm_feats)
                    ds_feats = feats[ds_train_mask].detach()
                    # ds_feats = F.normalize(ds_feats, dim=1)
                    ds_ys = ys[ds_train_mask]
                    y_hats = self.multi_classifier(ds_feats, i)
                    ds_classifier_loss = F.cross_entropy(y_hats, ds_ys)
                    classifier_loss += ds_classifier_loss

                    correct = torch.sum(torch.max(y_hats, 1)[1] == ds_ys)
                    total_correct += correct
            # loss_multi = self.multi_classifier
            total_loss = (classifier_loss)/self.train_cfg.dataset_number
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            tot_loss += total_loss
        iters += iters +1


        return total_correct / len(self.label_info.label_ind), tot_loss / iters

    def cal_train_inds(self, feats, ds_ind, target_ds_ind, sim_thrs, l_norm_feats):
        ds_mask = ds_ind==target_ds_ind
        tmp_mask = ds_ind!=target_ds_ind
        ds_train_mask = torch.zeros(len(feats), dtype=torch.bool).cuda() | ds_mask

        norm_feats = F.normalize(feats.detach(), dim=1)
        target_ds_feats = l_norm_feats[target_ds_ind]
        target_sim_thrs = sim_thrs[target_ds_ind]
        sim = torch.mm(norm_feats[~ds_mask], target_ds_feats.T)
        sorted_results = torch.max(sim, dim=1)
        sorted_values = sorted_results.values
        sorted_ind = sorted_results.indices
        ds_sim_thr = target_sim_thrs[sorted_ind]
        mask = sorted_values > ds_sim_thr
        ds_train_mask = torch.masked_scatter(ds_train_mask, tmp_mask, mask)
        return ds_train_mask


    def combine_train_each_epoch(self, optimizer, train_loader, epoch, name):
        self.net.train()
        accFinal, tot_loss, iters = 0., 0., 0

        net_opit = optimizer[0]
        dis_opit = optimizer[1]

        for Xs, ys, ind, ds_ind in train_loader:
            Xs = Xs.cuda()
            ys = ys.cuda()
            ds_ind = ds_ind.cuda()
            with torch.enable_grad():
                y_hats, feats = self.net(Xs)
                disc_input = ReverseLayerF.apply(feats, self.train_cfg.alpha)
                disc_out = self.domain_discriminator(disc_input)
                disc_loss = F.cross_entropy(disc_out, ds_ind)
                classifier_loss = F.cross_entropy(y_hats, ys)
                total_losses = disc_loss + classifier_loss

            accFinal += torch.sum((torch.max(y_hats, 1)[1] == ys).float()).data.item()

            net_opit.zero_grad()
            dis_opit.zero_grad()
            total_losses.backward()
            net_opit.step()
            dis_opit.step()


        iters += 1
        tot_loss += total_losses
        # print('epoch %d training loss %.4f %.4f'%(epoch, loss_ct, loss_ce))
        self.writer.add_scalar('training_loss/%s' % name, tot_loss / iters, epoch)

        return accFinal / len(self.label_info.label_ind), tot_loss / iters

    def get_center_weight(self, epoch):
        # center_weight = self.train_cfg['center_weights'][0]
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

    def get_sim_thr(self):
        label_loader = self.build_train_label_loader(shuffle=False)
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


