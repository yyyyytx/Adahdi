from .base_strategy import BaseStrategy
import numpy as np
import torch
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F
from scipy import stats
from utils import *

class CIRMSampling(BaseStrategy):

    def query(self, n_select):
        centers = self.trainer.center_loss.centers.detach().cpu()
        unlabel_loader, unlabel_ind = self.build_unlabel_loader()
        label_loader, label_ind = self.build_labeled_loader()
        u_embedding_features, u_true_labels, u_ds_inds, u_pred_cls, u_logits = self.predict_probs_and_embed(unlabel_loader)
        l_embedding_features, l_true_labels, l_ds_inds, l_pred_cls, l_logits = self.predict_probs_and_embed(label_loader)

        # region = torch.zeros()

        # print(u_embedding_features.shape)
        # print(u_ds_inds)
        u_embedding_features = F.normalize(u_embedding_features, dim=1)
        l_embedding_features = F.normalize(l_embedding_features, dim=1)
        u_logits = F.normalize(u_logits, dim=1)
        l_logits = F.normalize(l_logits, dim=1)
        u_data_ind = torch.arange(len(u_embedding_features))
        #
        # total_center = []
        # calculate center_radii
        # center_radii = torch.zeros(self.label_info.l_train_ds_number, self.net.n_label)
        # for ds_ind in range(self.label_info.l_train_ds_number):
        #     ds_center = F.normalize(centers[ds_ind], dim=1)
        #     l_mask = l_ds_inds == ds_ind
        #     l_ds_dist = torch.cdist(l_embedding_features[l_mask], ds_center)
        #     l_ds_labels = l_true_labels[l_mask]
        #     print(l_ds_labels.unsqueeze(dim=0).T.shape, l_ds_dist.shape)
        #
        #     l_dist = torch.gather(l_ds_dist, dim=1, index=l_ds_labels.unsqueeze(dim=0).T)
        #     l_dist = torch.squeeze(l_dist)
        #
        #     for cls in range(self.net.n_label):
        #         tmp_dist = l_dist[l_ds_labels==cls]
        #         if len(tmp_dist) == 0:
        #             center_radii[ds_ind][cls] = torch.tensor(0.)
        #         else:
        #             v = torch.max(tmp_dist)
        #             center_radii[ds_ind][cls] = v



        # l_mask = torch.zeros((len(l_embedding_features), self.net.n_label), dtype=torch.bool)
        # l_mask = torch.scatter(l_mask, 1, l_true_labels.unsqueeze(dim=0).T, True)
        # l_true_dist = []
        # l_false_dist = []
        # u_dist = []
        # for ds_ind in range(self.label_info.l_train_ds_number):
        #     ds_center = centers[ds_ind]
        #     l_ds_dist = torch.cdist(l_embedding_features, ds_center)
        #     u_ds_dist = torch.cdist(u_embedding_features, ds_center)
        #
        #     l_ds_ture_dist = l_ds_dist * l_mask
        #     l_ds_false_dist = l_ds_dist * (~l_mask) + l_mask * 9999.
        #     ds_true_dist = torch.max(l_ds_ture_dist, dim=1).values.unsqueeze(dim=0)
        #     ds_false_dist = torch.min(l_ds_false_dist, dim=1).values.unsqueeze(dim=0)
        #
        #     l_true_dist.append(ds_true_dist)
        #     l_false_dist.append(ds_false_dist)
        #
        #     u_dist.append(u_ds_dist.unsqueeze(dim=0))
        #
        # l_true_dist = torch.cat(l_true_dist, dim=0)
        # l_false_dist = torch.cat(l_false_dist, dim=0)
        # u_dist =torch.cat(u_dist, dim=0)
        # u_dist = torch.min(u_dist, dim=0).values
        #
        #
        # min_true_dist = torch.min(l_true_dist, dim=0).values
        # min_false_dist = torch.min(l_false_dist, dim=0).values
        # delta_dist = min_false_dist - min_true_dist
        #
        # cls_dist = []
        # for cls in range(self.net.n_label):
        #     tmp_dist = delta_dist[l_true_labels == cls]
        #     if len(tmp_dist) == 0:
        #         v = torch.tensor(0.)
        #     else:
        #         # v = torch.mean(tmp_dist)
        #         v = stats.mode(tmp_dist.numpy())[0][0]
        #         # print(tmp_dist.numpy())
        #         # print(v)
        #     cls_dist.append(v)
        # cls_dist = torch.tensor(cls_dist)
        # # print(cls_dist)
        #
        # sort = torch.sort(u_dist, descending=False)
        # u_dist_indices = sort.indices
        # u_dist_values = sort.values
        # u_delta_dist = u_dist_values[:, 1] - u_dist_values[:, 0]
        # # print(u_delta_dist[:10])
        # u_dist_t = cls_dist[u_dist_indices[:,0]]
        # mask = u_delta_dist > u_dist_t











        # print(torch.argsort(u_dist, descending=False))

        # u_dist = torch.min(u_dist, dim=0).values
        # u_dist = torch.min(u_dist, dim=0).values
        # u_dist_value, u_dist_indices = torch.min(u_dist, dim=1)
        # torch.topk(u_dist)


        # total_center = torch.cat(total_center, dim=0)
        # l_true_labels = l_true_labels.repeat((self.label_info.l_train_ds_number))
        # # print(total_center.shape)
        # l_feature_dist = torch.cdist(l_embedding_features, total_center)
        # l_dist_sort = torch.argsort(l_feature_dist, dim=1)
        # print(l_feature_dist.shape)
        # print(l_dist_sort)
        #
        #
        # feature_dist = torch.cdist(u_embedding_features, total_center)
        # u_dist_sort = torch.argsort(feature_dist, dim=0)
        # print(feature_dist.shape)
        # min_dist = torch.min(feature_dist, dim=1).values
        # true_dist = torch.gather(feature_dist, dim=1, index=u_true_labels.unsqueeze(dim=0).T)
        # true_dist = torch.squeeze(true_dist)
        # print(min_dist, true_dist)
        # farthest = torch.sort(min_dist, descending=True).indices[:n_select]
        # return unlabel_ind[farthest]




        mask = torch.zeros((len(u_embedding_features)), dtype=torch.bool)
        next_mask = torch.ones((len(u_embedding_features)), dtype=torch.bool)
        # u_indices = []

        #labeled ds selection
        for ds_ind in range(self.label_info.l_train_ds_number):
            ds_center = F.normalize(centers[ds_ind], dim=1)

            l_mask = l_ds_inds == ds_ind
            l_ds_logits = l_logits[l_mask]
            print(l_ds_logits.shape)
            l_ds_logits = torch.topk(l_ds_logits, k=2,dim=-1).values
            delta_logits = l_ds_logits[:,0]-l_ds_logits[:,1]
            print(torch.sort(delta_logits, dim=-1, descending=True))
            exit()

            l_ds_dist = torch.cdist(l_embedding_features[l_mask], ds_center, compute_mode='donot_use_mm_for_euclid_dist')
            l_ds_top = torch.topk(l_ds_dist, k=2, dim=1).values
            l_ds_margin = l_ds_top[0] - l_ds_top[1]


            u_ds_mask = u_ds_inds == ds_ind
            u_ds_dist = torch.cdist(u_embedding_features[u_ds_mask], ds_center,
                                    compute_mode='donot_use_mm_for_euclid_dist')
            u_ds_top = torch.topk(u_ds_dist, k=2, dim=1).values
            u_ds_margin = u_ds_top[0] - u_ds_top[1]
            m = u_ds_margin < 1.
            ds_mask = m & u_ds_mask
            print('select ds %d: %d' % (ds_ind, torch.sum(ds_mask)))

            mask = mask | ds_mask


        # labeled ds selection
        # for ds_ind in range(self.label_info.l_train_ds_number):
        #     ds_center = F.normalize(centers[ds_ind], dim=1)
        #     l_mask = l_ds_inds == ds_ind
        #     l_ds_dist = torch.cdist(l_embedding_features[l_mask], ds_center, compute_mode='donot_use_mm_for_euclid_dist')
        #     l_ds_labels = l_true_labels[l_mask]
        #     l_dist = torch.gather(l_ds_dist, dim=1, index=l_ds_labels.unsqueeze(dim=0).T)
        #     print(l_ds_dist[:2])
        #     print(l_ds_labels[:2])
        #     l_dist = torch.squeeze(l_dist)
        #     print(l_dist[:2])
        #     # print('l_dist shape:', l_dist, ds_center.shape, l_ds_dist.shape)
        #     print('center dist:', torch.cdist(ds_center, ds_center, compute_mode='donot_use_mm_for_euclid_dist'))
        #
        #     cls_dist = []
        #     for cls in range(self.net.n_label):
        #         tmp_dist = l_dist[l_ds_labels==cls]
        #         print(tmp_dist)
        #         if len(tmp_dist) == 0:
        #             v = torch.tensor(0.)
        #         else:
        #             v = torch.max(tmp_dist)
        #             # v = stats.mode(tmp_dist.numpy())[0][0]
        #             #  v = torch.mean(tmp_dist)
        #         v = 0.5
        #         cls_dist.append(v)
        #     cls_dist = torch.tensor(cls_dist)
        #     print(cls_dist)
        #
        #     ds_dist = torch.cdist(u_embedding_features, ds_center , compute_mode='donot_use_mm_for_euclid_dist')
        #
        #     # check_u_mask = u_ds_inds == ds_ind
        #     # check_u_true_label = u_true_labels[check_u_mask]
        #     # flag = u_dist < cls_dist_t
        #
        #     dist_value, dist_indices = torch.min(ds_dist, dim=1)
        #     # print(dist_value)
        #
        #     # u_indices.append(dist_indices)
        #     t = cls_dist[dist_indices]
        #     m = dist_value > t
        #     u_ds_mask = u_ds_inds == ds_ind
        #     ds_mask = m & u_ds_mask
        #     print('select ds %d: %d' % (ds_ind, torch.sum(ds_mask)))
        #
        #     mask = mask | ds_mask
        #
        #     next_u_ds_mask = u_ds_inds == self.label_info.l_train_ds_number
        #     print(torch.sum(next_u_ds_mask))
        #     next_ds_mask = m & next_u_ds_mask
        #     next_mask = next_ds_mask & next_mask
        #     # next_m = next_ds_mask & next_mask
        #     # mask = mask & next_mask
        # mask = mask | next_mask
        # print('select next ds %d' % torch.sum(next_mask))
            # print('ds_ind:', ds_ind)
            #
            # ds_larger_correct = torch.sum(u_pred_cls[ds_mask] == u_true_labels[ds_mask])
            # print('select correct: %d/%d' % (ds_larger_correct, torch.sum(m)))
            # ds_smaller_correct = torch.sum(u_pred_cls[u_ds_mask & ~m] == u_true_labels[u_ds_mask & ~m])
            # print('not select correct: %d/%d' % (ds_smaller_correct, torch.sum(~m)))

        # unlabeled ds selection
        # u_ds_ind = self.label_info.l_train_ds_number+1



        # calculate centers
        # centers = []
        # for cls in range(self.net.n_label):
        #     # print(torch.mean(l_embedding_features[l_true_labels == cls], dim=0).shape)
        #     centers.append(torch.mean(l_embedding_features[l_true_labels == cls], dim=0).unsqueeze(dim=0))
        # centers = torch.cat(centers, dim=0)
        # l_dist = torch.cdist(l_embedding_features, centers)
        # cls_dist_t = []
        # for cls in range(self.net.n_label):
        #     cls_dist = l_dist[:, cls]
        #     cls_mask = l_true_labels == cls
        #     cls_true_dist = cls_dist[cls_mask]
        #     cls_false_dist = cls_dist[~cls_mask]
        #     min_false = torch.min(cls_false_dist)
        #     # print(cls_true_dist)
        #     # print(min_false)
        #     tmp_flag = cls_true_dist < min_false
        #     if torch.sum(tmp_flag) == 0:
        #         v = min_false
        #     else:
        #         v = torch.max(cls_true_dist[tmp_flag])
        #
        #     # max_cls_true_dist = torch.max(cls_true_dist[tmp_flag])
        #     cls_dist_t.append(v)
        # cls_dist_t = torch.tensor(cls_dist_t)
        # # print(cls_dist_t)
        # cls_dist_t = torch.unsqueeze(cls_dist_t, dim=0).repeat((len(u_embedding_features), 1))
        #
        # u_dist = torch.cdist(u_embedding_features, centers)
        # center_dist = torch.cdist(centers, centers)
        # flag = u_dist < cls_dist_t
        # flag_sum = torch.sum(flag, dim=1)
        #
        # confident_mask = flag_sum <= 1
        #
        # flag1 = flag_sum == 1
        # flag1_correct = torch.sum(u_pred_cls[flag1] == u_true_labels[flag1])
        # print('flag1 correct: %d/%d' % (flag1_correct, torch.sum(flag1)))
        #
        # # print('ds_ind : %d flag1 correct: %d/%d' % (flag1_correct, torch.sum(flag1)))
        #
        # flag0 = flag_sum == 0
        # flag0_correct = torch.sum(u_pred_cls[flag0] == u_true_labels[flag0])
        # print('flag0 correct: %d/%d' % (flag0_correct, torch.sum(flag0)))
        #
        # flag2 = flag_sum > 1
        # flag2_correct = torch.sum(u_pred_cls[flag2] == u_true_labels[flag2])
        # print('flag2 correct: %d/%d' % (flag2_correct, torch.sum(flag2)))
        # for ds_ind in range(self.label_info.l_train_ds_number+1):
        #     print('ds_ind:', ds_ind)
        #     ds_flag1 = flag1 & (u_ds_inds == ds_ind)
        #     ds_flag1_correct = torch.sum(u_pred_cls[ds_flag1] == u_true_labels[ds_flag1])
        #     print('     flag1 correct: %d/%d' % (ds_flag1_correct, torch.sum(flag1[ds_flag1])))
        #     ds_flag0 = flag0 & (u_ds_inds == ds_ind)
        #     ds_flag0_correct = torch.sum(u_pred_cls[ds_flag0] == u_true_labels[ds_flag0])
        #     print('     flag0 correct: %d/%d' % (ds_flag0_correct, torch.sum(flag0[ds_flag0])))
        #     ds_flag2 = flag2 & (u_ds_inds == ds_ind)
        #     ds_flag2_correct = torch.sum(u_pred_cls[ds_flag2] == u_true_labels[ds_flag2])
        #     print('     flag2 correct: %d/%d' % (ds_flag2_correct, torch.sum(flag2[ds_flag2])))
        # mask = ~confident_mask

        coreset_embedding = u_embedding_features[mask]
        coreset_ind = u_data_ind[mask]
        # coreset_embedding = torch.cat(coreset_embedding, dim=0).cpu()
        # coreset_ind = torch.cat(coreset_ind, dim=0).cpu()
        query = kCenterGreedy(features=coreset_embedding)
        str = 'coreset len: %d' % len(coreset_embedding)
        print(str)
        self.trainer.logger.info(str)
        if len(coreset_embedding) < n_select:
            tmp_ind = np.random.permutation(u_data_ind[~mask])[:n_select-len(coreset_embedding)]
            str = 'random select %d' % len(tmp_ind)
            print(str)
            self.trainer.logger.info(str)
            return unlabel_ind[np.append(coreset_ind, tmp_ind)]
        else:
            select_ind = query.select_batch_(already_selected=[], N=n_select)
            select_ds_ind = u_ds_inds[coreset_ind[select_ind]]
            select_ds_count = []
            for ds_ind in range(self.label_info.l_train_ds_number):
                select_ds_count.append(torch.sum(select_ds_ind == ds_ind))
            str = 'select ds count:'
            self.trainer.logger.info(str)
            self.trainer.logger.info(select_ds_count)
            print(str)
            print(select_ds_count)
            return unlabel_ind[coreset_ind[select_ind]]

        # print(coreset_embedding.shape)
        # print(coreset_ind.shape)
        # print(len(select_ind))

        # _, unlabel_ind = self.build_unlabel_loader()
        # tmp_ind = np.random.permutation(range(len(unlabel_ind)))[:n_select]

    def predict_probs_and_embed(self, data_loader, eval=True):
        self.net.eval()
        # probs = []
        logits = []
        embedding_features = []
        ds_inds = []
        true_labels = []
        pred_cls = []


        if eval:
            self.net.eval()
        else:
            self.net.train()

        for x, y, ind, ds_ind in data_loader:
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                out, e1 = self.net(x)
                _, preds = torch.max(out, 1)

            # prob = F.softmax(out, dim=1)
            # probs[idxs] = prob.cpu()
            # embeddings[idxs] = e1.cpu()

            logits.append(out)
            true_labels.append(y)
            pred_cls.append(preds)

            embedding_features.append(e1)
            ds_inds.append(ds_ind)
            # probs.append(prob)

        logits = torch.cat(logits, dim=0).cpu()
        # probs = torch.cat(probs, dim=0).cpu()
        pred_cls = torch.cat(pred_cls,dim=0).cpu()

        embedding_features = torch.cat(embedding_features, dim=0).cpu()
        true_labels = torch.cat(true_labels, dim=0).cpu()
        ds_inds = torch.cat(ds_inds, dim=0)

        return embedding_features, true_labels, ds_inds, pred_cls, logits