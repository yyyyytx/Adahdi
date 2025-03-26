from .base_trainer import BaseTrainer
import torch
from tqdm import tqdm
from trainer.losses import CenterSeperateMarginLoss
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import time
from models.ltc_module import GCN
from torch import nn

class LtCMSDATrainer(BaseTrainer):
    def __init__(self, net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg=None, is_amp=False, logger=None):
        super().__init__(net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg, is_amp, logger)
        self.net = net[0]
        self.GCN = GCN(nfeat=self.net.get_embedding_dim(), nclasses=self.net.n_label)


    def train(self, name, n_epoch=None):
        train_loader = self.build_train_label_loader()
        self.ndomain = self.train_cfg.dataset_number

        opti, lr_sched = self.build_optimizer(self.net)
        gcn_opti, gcn_sched = self.build_optimizer(self.GCN)

        self.mean = torch.zeros(self.net.n_label * self.ndomain, self.net.get_embedding_dim()).cuda()
        self.adj = torch.zeros(self.net.n_label * self.ndomain, self.net.n_label * self.ndomain).cuda()

        acclist = []
        best_acc = 0.
        for epoch in tqdm(range(self.train_cfg.epochs)):
            accTrain, lossTrain = self.train_epoch_train([opti, gcn_opti], train_loader, epoch, name)
            # self.writer.add_scalar('training_accuracy/%s' % name, accTrain, epoch)
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

    def update_statistics(self, feats, labels, epsilon=1e-5):
        curr_mean = list()
        num_labels = 0

        for domain_idx in range(self.ndomain):
            tmp_feat = feats[domain_idx]
            tmp_label = labels[domain_idx]
            num_labels += tmp_label.shape[0]

            if tmp_label.shape[0] == 0:
                curr_mean.append(torch.zeros((self.net.n_label, self.net.get_embedding_dim())).cuda())
            else:
                onehot_label = torch.zeros((tmp_label.shape[0], self.net.n_label)).scatter_(1, tmp_label.unsqueeze(
                    -1).cpu(), 1).float().cuda()
                domain_feature = tmp_feat.unsqueeze(1) * onehot_label.unsqueeze(-1)
                tmp_mean = domain_feature.sum(0) / (onehot_label.unsqueeze(-1).sum(0) + epsilon)

                curr_mean.append(tmp_mean)

        curr_mean = torch.cat(curr_mean, dim=0)
        curr_mask = (curr_mean.sum(-1) != 0).float().unsqueeze(-1)
        self.mean = self.mean.detach() * (1 - curr_mask) + (
                self.mean.detach() * 0.7 + curr_mean * (1 - 0.7)) * curr_mask
        curr_dist = self.euclid_dist(self.mean, self.mean)
        self.adj = torch.exp(-curr_dist / (2 * 0.005 ** 2))

        # compute local relation alignment loss
        loss_local = ((((curr_mean - self.mean) * curr_mask) ** 2).mean(-1)).sum() / num_labels

        return loss_local

    def euclid_dist(self, x, y):
        x_sq = (x ** 2).mean(-1)
        x_sq_ = torch.stack([x_sq] * y.size(0), dim=1)
        y_sq = (y ** 2).mean(-1)
        y_sq_ = torch.stack([y_sq] * x.size(0), dim=0)
        xy = torch.mm(x, y.t()) / x.size(-1)
        dist = x_sq_ + y_sq_ - 2 * xy

        return dist

    def construct_adj(self, means, feats):
        dist = self.euclid_dist(means, feats)
        sim = torch.exp(-dist / (2 * 0.005 ** 2))
        E = torch.eye(feats.shape[0]).float().cuda()

        A = torch.cat([self.adj, sim], dim=1)
        B = torch.cat([sim.t(), E], dim=1)
        gcn_adj = torch.cat([A, B], dim=0)

        return gcn_adj

    def adj_loss(self):
        adj_loss = 0

        for i in range(self.ndomain):
            for j in range(self.ndomain):
                adj_ii = self.adj[i * self.net.n_label:(i + 1) * self.net.n_label,
                         i * self.net.n_label:(i + 1) * self.net.n_label]
                adj_jj = self.adj[j * self.net.n_label:(j + 1) * self.net.n_label,
                         j * self.net.n_label:(j + 1) * self.net.n_label]
                adj_ij = self.adj[i * self.net.n_label:(i + 1) * self.net.n_label,
                         j * self.net.n_label:(j + 1) * self.net.n_label]

                adj_loss += ((adj_ii - adj_jj) ** 2).mean()
                adj_loss += ((adj_ij - adj_ii) ** 2).mean()
                adj_loss += ((adj_ij - adj_jj) ** 2).mean()
        if self.ndomain > 1:
            adj_loss /= (self.ndomain * (self.ndomain - 1) / 2 * 3)

        return adj_loss

    def train_epoch_train(self, optimizer, train_loader, epoch, name):

        self.net.train()
        self.GCN.train()
        accFinal, tot_loss, iters = 0., 0., 0
        net_opit = optimizer[0]
        gcn_opit = optimizer[1]
        criterion = nn.CrossEntropyLoss().cuda()

        for Xs, ys, ind, ds_ind in train_loader:
            img_s = list()
            label_s = list()
            stop_iter = False

            for domain_idx in range(self.ndomain):
                tmp_img = Xs[domain_idx == ds_ind].cuda()
                tmp_label = ys[domain_idx == ds_ind].long().cuda()
                img_s.append(tmp_img)
                label_s.append(tmp_label)

            net_opit.zero_grad()
            gcn_opit.zero_grad()

            feats = list()
            for domain_idx in range(self.ndomain):
                tmp_img = img_s[domain_idx]
                _, tmp_feat = self.net(tmp_img)
                feats.append(tmp_feat)

            # Update the global mean and adjacency matrix
            loss_local = self.update_statistics(feats, label_s)
            feats = torch.cat(feats, dim=0)
            labels = torch.cat(label_s, dim=0)

            # add query samples to the domain graph
            gcn_feats = torch.cat([self.mean, feats], dim=0)
            gcn_adj = self.construct_adj(self.mean, feats)

            # output classification logit with GCN
            gcn_logit = self.GCN(gcn_feats, gcn_adj)

            # define GCN classification losses
            domain_logit = gcn_logit[:self.mean.shape[0], :]
            domain_label = torch.cat([torch.arange(self.net.n_label)] * self.ndomain, dim=0)
            domain_label = domain_label.long().cuda()
            loss_cls_dom = criterion(domain_logit, domain_label)

            query_logit = gcn_logit[self.mean.shape[0]:, :]
            loss_cls_src = criterion(query_logit, labels)

            loss_cls = loss_cls_src + loss_cls_dom
            # define relation alignment losses
            loss_global = self.adj_loss() * 20
            loss_local = loss_local * 20
            loss_relation = loss_local + loss_global

            loss = loss_cls + loss_relation

            loss.backward()
            net_opit.step()
            gcn_opit.step()

        iters += 1
        tot_loss += loss
        return accFinal / len(self.label_info.label_ind), tot_loss / iters

    def base_model_accuracy(self):
        self.net.eval()
        self.GCN.eval()
        loader = self.build_test_loader()
        correct_num = 0

        ds_inds = []
        pred_cls = []
        true_labels = []

        test_loss = 0
        correct = 0
        size = 0

        for Xs, ys, ind, ds_ind in tqdm(loader):
            img = Xs
            label = ys
            img, label = img.cuda(), label.long().cuda()

            _, feat = self.net(img)
            gcn_adj = self.construct_adj(self.mean, feat)
            gcn_feats = torch.cat([self.mean, feat], dim=0)
            gcn_logit = self.GCN(gcn_feats, gcn_adj)


            output = gcn_logit[self.mean.shape[0]:, :]

            pred = output.max(1)[1]

            del gcn_feats
            del gcn_adj
            del gcn_logit
            del output
            correct_num += pred.eq(label).cpu().sum()
            # print(correct_num)

            pred_cls.append(pred)
            true_labels.append(label)
            ds_inds.append(ds_ind)



            # k = label.size()[0]
            # correct += pred.eq(label).cpu().sum()
            # size += k
        pred_cls = torch.cat(pred_cls, dim=0)
        ds_inds = torch.cat(ds_inds, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        correct_mask = pred_cls == true_labels
        result = {}
        result['total'] = (float(correct_num) / float(len(loader.dataset)))
        for i in range(len(self.test_ds.domain_lens)):
            key = 'ds' + str(i)
            value = torch.sum(correct_mask[ds_inds == i]) / float(self.test_ds.domain_lens[i])
            result[key] = value
        mask = ds_inds < self.train_cfg.dataset_number
        result['current'] = torch.sum(correct_mask[mask]) / float(torch.sum(mask))
        return result


        # test_loss = test_loss / size
        #
        # # record test information
        # print(
        #     '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), Best Accuracy: {}/{} ({:.4f}%)  \n'.format(
        #         test_loss, correct, size, 100. * float(correct) / size, self.best_correct, size,
        #                                   100. * float(self.best_correct) / size))