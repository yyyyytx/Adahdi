from .base_trainer import BaseTrainer
import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import time

class TiDALTrainer(BaseTrainer):
    def __init__(self, net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg=None, is_amp=False, logger=None):
        super().__init__(net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg, is_amp, logger)
        self.pred_module = self.net[1]
        self.net = self.net[0]

    def train(self, name, n_epoch=None):
        opti, lr_sched = self.build_optimizer(self.net)
        module_opti, module_sched = self.build_optimizer(self.pred_module)
        optimizers = {'backbone': opti, 'module': module_opti}
        # schedulers = {'backbone': lr_sched, 'module': module_sched}

        train_loader = self.build_train_label_loader()
        acclist = []
        best_acc = 0.
        for epoch in tqdm(range(self.train_cfg.epochs)):
            accTrain, losses = self.train_each_epoch(self.net, self.pred_module, optimizers, train_loader, epoch, name)
            if lr_sched is not None:
                lr_sched.step(epoch)
            if module_sched is not None:
                module_sched.step(epoch)
            acclist.append(accTrain)

            if (epoch + 1) % self.train_cfg.val_interval == 0:
                current_acc = self.base_model_accuracy()['current']
                cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                str = cur_time + 'train epoch: %d  train acc %.4f cur acc %.4f loss: %.4f' % (
                epoch + 1, accTrain, current_acc, losses)
                print(str)
                self.logger.info(str)
                self.writer.add_scalar('test_accuracy/%s' % name, current_acc, epoch)
                if current_acc > best_acc:
                    best_acc = current_acc
        return best_acc


    def train_each_epoch(self, net, pred_module, optimizers, dataloader, epoch, name):
        net.train()
        pred_module.train()
        accFinal, tot_loss, iters = 0., 0., 0
        true_labels = []
        ce = nn.CrossEntropyLoss(reduction='none')
        kl = nn.KLDivLoss(reduction='batchmean')

        for Xs, ys, ind, _, moving_prob in dataloader:
            inputs = Xs.cuda()
            labels = ys.cuda()
            index = ind.detach().numpy().tolist()
            moving_prob = moving_prob.cuda()
            true_labels.append(ys)

            optimizers['backbone'].zero_grad()
            optimizers['module'].zero_grad()

            scores, emb, features = self.net.forward_features(inputs)
            target_loss = ce(scores, labels)
            probs = torch.softmax(scores, dim=1)

            moving_prob = (moving_prob * epoch + probs * 1) / (epoch + 1)
            # print(dataloader.dataset.dataset)
            dataloader.dataset.dataset.moving_prob[index, :] = moving_prob.cpu().detach().numpy()
            cumulative_logit = pred_module(features)
            m_module_loss = kl(F.log_softmax(cumulative_logit, 1), moving_prob.detach())
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            loss = m_backbone_loss + self.train_cfg.WEIGHT * m_module_loss

            loss.backward()
            optimizers['backbone'].step()
            optimizers['module'].step()

            tot_loss += loss.item()
            iters += 1
            accFinal += torch.sum((torch.max(scores, 1)[1] == labels).float()).data.item()
        self.writer.add_scalar('training_loss/%s' % name, tot_loss / (iters + 1), epoch)
        return accFinal / len(dataloader.dataset), tot_loss / (iters + 1)


    def base_model_accuracy(self):
        self.net.eval()
        correct_num = 0
        ds_inds = []
        pred_cls = []
        true_labels = []

        loader = self.build_test_loader()
        for Xs, ys, ind, ds_ind, _ in tqdm(loader):
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



