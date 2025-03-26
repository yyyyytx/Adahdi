from .base_trainer import BaseTrainer
import torch
from tqdm import tqdm
from .losses import EDL_Loss
import time

class MADATrainer(BaseTrainer):
    def __init__(self, net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg=None, is_amp=False, logger=None):
        super().__init__(net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg, is_amp, logger)
        self.net = net[0]

    def train(self, name, n_epoch=None):

        opti, lr_sched = self.build_optimizer(self.net)

        train_loader = self.build_train_label_loader()
        unlabeled_loader = self.build_train_unlabel_loader()
        selected_loader = self.build_train_label_loader()

        acclist = []
        best_acc = 0.

        if n_epoch == None:
            epochs = self.train_cfg.epochs
        else:
            epochs = n_epoch

        for epoch in tqdm(range(self.train_cfg.epochs)):
            lossTrain = self.combine_train_each_epoch(opti, [train_loader, unlabeled_loader, selected_loader], epoch, name)
            print(lossTrain)
            if lr_sched is not None:
                lr_sched.step(epoch)
            if (epoch + 1) % self.train_cfg.val_interval == 0:
                result = self.base_model_accuracy()
                print(result)
                current_acc = result['current']
                cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                str = cur_time + 'train epoch: %d  cur acc %.4f loss %.4f' % (epoch + 1, current_acc, lossTrain)
                print(str)
                self.logger.info(str)
                self.writer.add_scalar('test_accuracy/%s' % name, current_acc, epoch)
                if current_acc > best_acc:
                    best_acc = current_acc

        return best_acc


    def combine_train_each_epoch(self, optimizer, train_loader, epoch, name):
        self.net.train()
        edl_criterion = EDL_Loss()
        accFinal, tot_loss, iters = 0., 0., 0

        net_opit = optimizer

        l_loader = train_loader[0]
        u_loader = train_loader[1]
        s_loader = train_loader[2]
        l_loader = iter(l_loader)
        joint_loader = zip(l_loader, u_loader, s_loader)

        for batch_idx, ((Xs_l, ys_l, _, ds_ind_l), (Xs_u, _, _, ds_ind_u), (Xs_s, ys_s, _, ds_ind_s)) in enumerate(joint_loader):
            total_loss = 0.

            Xs_l, ys_l, ds_ind_l = Xs_l.cuda(), ys_l.cuda(), ds_ind_l.cuda()
            Xs_u, ds_ind_u = Xs_u.cuda(), ds_ind_u.cuda()
            Xs_s, ys_s, ds_ind_s = Xs_s.cuda(), ys_s.cuda(), ds_ind_s.cuda()
            src_out = self.net.forward_mada(Xs_l, return_feat=False)
            Loss_nll_s, Loss_KL_s = edl_criterion(src_out, ys_l)
            Loss_KL_s = Loss_KL_s / self.net.n_label
            total_loss += Loss_nll_s
            total_loss += Loss_KL_s

            # default BETA=1.0
            if self.train_cfg.BETA > 0.:
                tgt_unlabeled_out = self.net.forward_mada(Xs_u, return_feat=False)
                alpha_t = torch.exp(tgt_unlabeled_out)
                total_alpha_t = torch.sum(alpha_t, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
                expected_p_t = alpha_t / total_alpha_t
                eps = 1e-7
                point_entropy_t = - torch.sum(expected_p_t * torch.log(expected_p_t + eps), dim=1)
                data_uncertainty_t = torch.sum(
                    (alpha_t / total_alpha_t) * (torch.digamma(total_alpha_t + 1) - torch.digamma(alpha_t + 1)), dim=1)
                loss_Udis = torch.sum(point_entropy_t - data_uncertainty_t) / tgt_unlabeled_out.shape[0]
                loss_Udata = torch.sum(data_uncertainty_t) / tgt_unlabeled_out.shape[0]

                total_loss += self.train_cfg.BETA * loss_Udis
                total_loss += self.train_cfg.LAMBDA * loss_Udata

            tgt_selected_out = self.net.forward_mada(Xs_s, return_feat=False)
            selected_Loss_nll_t, selected_Loss_KL_t = edl_criterion(tgt_selected_out, ys_s)
            selected_Loss_KL_t = selected_Loss_KL_t / self.net.n_label
            total_loss += selected_Loss_nll_t
            total_loss += selected_Loss_KL_t

            net_opit.zero_grad()
            total_loss.backward()
            net_opit.step()

            tot_loss += total_loss
            iters += 1

            # print(Loss_nll_s, Loss_KL_s, loss_Udis, loss_Udata, selected_Loss_nll_t, selected_Loss_KL_t)
        return tot_loss / iters


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
                y_hats = self.net.forward_mada(Xs, return_feat=False)
                # y_hats, feats = self.net(Xs)
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


