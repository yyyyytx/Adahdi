from .base_strategy import BaseStrategy
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans
import copy


class AlphaMixSampling(BaseStrategy):
    '''
    (CVPR2022)Active Learning by Feature Mixing
    '''
    def query(self, n_select):
        self.net = self.net[0]
        self.query_count += 1
        label_loader, label_ind = self.build_labeled_loader()
        unlabel_loader, unlabel_ind = self.build_unlabel_loader()

        ulb_probs, org_ulb_embedding, _, _ = self.predict_probs_and_embed(unlabel_loader)
        probs_sorted, probs_sort_idxs = ulb_probs.sort(descending=True)
        pred_1 = probs_sort_idxs[:, 0]

        lb_probs, org_lb_embedding, lb_y_list, _ = self.predict_probs_and_embed(label_loader)

        ulb_embedding = org_ulb_embedding
        lb_embedding = org_lb_embedding
        unlabeled_size = ulb_embedding.size(0)
        embedding_size = ulb_embedding.size(1)

        min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
        candidate = torch.zeros(unlabeled_size, dtype=torch.bool)

        if self.strategy_cfg.alpha_closed_form_approx:
            var_emb = Variable(ulb_embedding, requires_grad=True).cuda()
            out, _ = self.net(var_emb, embedding=True)
            loss = F.cross_entropy(out, pred_1.cuda())
            grads = torch.autograd.grad(loss, var_emb)[0].data.cpu()
            del loss, var_emb, out
        else:
            grads = None

        alpha_cap = 0.
        sub_size = self.active_cfg.sub_num
        while alpha_cap < 1.0:
            alpha_cap += self.strategy_cfg.alpha_cap
            print(ulb_embedding.shape, ulb_probs.shape, lb_embedding.shape, pred_1.shape)
            list_len = math.ceil(len(ulb_embedding) / sub_size)
            tmp_pred_change = []
            for i in range(list_len):
                print('unlabel loader: %d/%d' % (i + 1, list_len))

                if i == list_len - 1:
                    tmp_ulb_embedding = ulb_embedding[i*sub_size:]
                    tmp_pred_1 = pred_1[i*sub_size:]
                    tmp_ulb_probs = ulb_probs[i*sub_size:]
                    tmp_grads = grads[i*sub_size:]
                else:
                    tmp_ulb_embedding = ulb_embedding[i * sub_size: (i+1)*sub_size]
                    tmp_pred_1 = pred_1[i * sub_size: (i+1)*sub_size]
                    tmp_ulb_probs = ulb_probs[i * sub_size: (i+1)*sub_size]
                    tmp_grads = grads[i * sub_size: (i+1)*sub_size]

                    # unlabel_ind = self.label_info.unlabel_ind[i * loader_len:(i + 1) * loader_len]

                tmp_pred_change1, _ = \
                    self.find_candidate_set(
                        lb_embedding, tmp_ulb_embedding, tmp_pred_1, tmp_ulb_probs, alpha_cap=alpha_cap,
                        Y=lb_y_list,
                        grads=tmp_grads)
                tmp_pred_change.append(tmp_pred_change1)
            tmp_pred_change = torch.cat(tmp_pred_change, dim=0)
            print(tmp_pred_change.shape)
            # is_changed = min_alphas.norm(dim=1) >= tmp_min_alphas.norm(dim=1)

            # min_alphas[is_changed] = tmp_min_alphas[is_changed]
            candidate += tmp_pred_change
            if candidate.sum() > n_select:
                break

        if candidate.sum() > 0:
            c_alpha = F.normalize(org_ulb_embedding[candidate].view(candidate.sum(), -1), p=2, dim=1).detach()
            selected_idxs = self.sample(min(n_select, candidate.sum().item()), feats=c_alpha)
            u_selected_idxs = candidate.nonzero(as_tuple=True)[0][selected_idxs]
            selected_idxs = unlabel_ind[candidate][selected_idxs]
        else:
            selected_idxs = np.array([], dtype=np.int)

        if len(selected_idxs) < n_select:
            remained = n_select - len(selected_idxs)
            tmp_unlabed_ind = copy.deepcopy(unlabel_ind)
            remained_ind = np.setdiff1d(tmp_unlabed_ind, selected_idxs)
            random_ind = np.random.permutation(remained_ind)[:remained]
            selected_idxs = np.append(selected_idxs, random_ind)
            # print(random_ind, selected_idxs)
            # selected_idxs = np.concatenate([selected_idxs, np.random.choice(remained_ind, remained)])
            print('picked %d samples from RandomSampling.' % (remained))

        return selected_idxs

    def sample(self, n, feats):
        feats = feats.numpy()
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(feats)

        cluster_idxs = cluster_learner.predict(feats)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (feats - centers) ** 2
        dis = dis.sum(axis=1)
        return np.array(
            [np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n) if
             (cluster_idxs == i).sum() > 0])

    def find_candidate_set(self, lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap, Y, grads):
        unlabeled_size = ulb_embedding.size(0)
        embedding_size = ulb_embedding.size(1)

        min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
        pred_change = torch.zeros(unlabeled_size, dtype=torch.bool)

        if self.strategy_cfg.alpha_closed_form_approx:
            alpha_cap /= math.sqrt(embedding_size)
            grads = grads.cuda()
        # if self.args.alpha_closed_form_approx:
        for i in range(self.net.n_label):
            emb = lb_embedding[Y == i]
            if emb.size(0) == 0:
                emb = lb_embedding
            anchor_i = emb.mean(dim=0).view(1, -1).repeat(unlabeled_size, 1)

            if self.strategy_cfg.alpha_closed_form_approx:
                embed_i, ulb_embed = anchor_i.cuda(), ulb_embedding.cuda()
                alpha = self.calculate_optimum_alpha(alpha_cap, embed_i, ulb_embed, grads)
                # print(embed_i.shape, ulb_embed.shape, alpha.shape)
                embedding_mix = (1 - alpha) * ulb_embed + alpha * embed_i
                out, _ = self.net(embedding_mix.cuda(), embedding=True)
                out = out.detach().cpu()
                alpha = alpha.cpu()

                pc = out.argmax(dim=1) != pred_1


            torch.cuda.empty_cache()
            # self.writer.add_scalar('stats/inconsistencies_%d' % i, pc.sum().item(), self.query_count)

            alpha[~pc] = 1.
            pred_change[pc] = True
            is_min = min_alphas.norm(dim=1) > alpha.norm(dim=1)
            min_alphas[is_min] = alpha[is_min]

            # self.writer.add_scalar('stats/inconsistencies_%d' % i, pc.sum().item(), self.query_count)

        return pred_change, min_alphas

    def calculate_optimum_alpha(self, eps, lb_embedding, ulb_embedding, ulb_grads):
        # print(lb_embedding.shape, ulb_embedding.shape, ulb_grads.shape)
        z = (lb_embedding - ulb_embedding)  # * ulb_grads
        ulb_grads = ulb_grads
        alpha = (eps * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1).repeat(1, z.size(1)) * ulb_grads / (
                    z + 1e-8)

        return alpha