import copy

from .base_strategy import BaseStrategy
import numpy as np
import torch.nn.functional as F
import torch

class TMSSampling(BaseStrategy):

    def query(self, n_select):
        print('start AdaHDI')

        self.multi_classifier = self.trainer.multi_classifier
        self.n_multi_classfiers = len(self.multi_classifier.multi_classifiers)
        self.net = self.trainer.net
        # self.center_sim_thrs = self.trainer.get_center_sim_thr().cuda()
        self.center_sim_thrs = self.trainer.cal_domain_sim_thr().cuda()
        centroids = self.trainer.center_loss.centers


        loader_list = self.build_divided_unlabel_loader(self.active_cfg.sub_num)
        delta = []
        for i, unlabel_loader in enumerate(loader_list):
            print('unlabel loader: %d/%d' % (i + 1, len(loader_list)))
            tmp_delta = self.cal_domain_margin1(unlabel_loader, centroids)
            delta.append(tmp_delta)
        delta = torch.cat(delta, dim=0)

        # unlabel_loader, unlabel_ind = self.build_sequence_unlabel_loader()
        # delta = self.cal_domain_margin(unlabel_loader, centroids)


        # print(total_margin.shape)

        # probs = self.predict_multi(unlabel_loader, centroids)
        # pros = torch.topk(probs, k=2, dim=1).values
        # delta = pros[:, 0] - pros[:, 1]
        #
        margin_ind = torch.argsort(delta, descending=True)[-n_select:].cpu()
        select_ind = self.label_info.unlabel_ind[margin_ind]

        return select_ind



    def cal_weighted_domain_margin(self, data_loader, centroids):
        self.net.eval()
        self.multi_classifier.eval()
        total_margins = []
        for x, _, ind, ds_ind in data_loader:
            x, ds_ind = x.cuda(), ds_ind.cuda()

            with torch.no_grad():
                out, feats = self.net(x)

            known_domain_mask = (ds_ind < self.n_multi_classfiers)
            unknown_domain_mask = (ds_ind == self.n_multi_classfiers)

            if torch.sum(known_domain_mask) != 0:
                multi_domain_preds = []
                multi_domain_margins = []
                multi_domain_weights = []
                for i in range(self.n_multi_classfiers):
                    preds = self.multi_classifier(feats[known_domain_mask], i)
                    multi_domain_preds.append(preds)
                for i in range(self.n_multi_classfiers):
                    ds_mask = ds_ind[known_domain_mask] == i
                    ds_weights = torch.zeros_like(ds_mask, dtype=torch.float)
                    ds_weights = torch.masked_fill(ds_weights, ds_mask, 1.)


                    ds_centroids = F.normalize(centroids[i], dim=1)
                    ds_norm_feats = F.normalize(feats[known_domain_mask][~ds_mask], dim=1)
                    sims = torch.mm(ds_norm_feats, ds_centroids.T)

                    target_sim_thrs = self.center_sim_thrs[i]
                    target_sim_thrs = target_sim_thrs.repeat((len(sims), 1)).cuda()
                    thr_mask = sims > target_sim_thrs
                    # prob_sims = torch.zeros_like(thr_mask, dtype=torch.float)
                    # print(sims)
                    # print(thr_mask)

                    tmp_probs = abs((1 - (1 - sims) / (1-target_sim_thrs+0.00001)) * thr_mask)


                    # prob_sims = torch.masked_scatter(prob_sims, thr_mask, sims[thr_mask])
                    # print(i, prob_sims)
                    prob_sims = torch.max(tmp_probs, dim=1).values


                    # prob_sims = torch.max(sims, dim=1).values
                    ds_weights = torch.masked_scatter(ds_weights, ~ds_mask, prob_sims)
                    multi_domain_weights.append(ds_weights.unsqueeze(0))


                    probs = F.softmax(multi_domain_preds[i], dim=1)
                    probs = torch.topk(probs, k=2, dim=1).values
                    margin = probs[:, 0] - probs[:, 1]
                    multi_domain_margins.append(margin.unsqueeze(0))

                multi_domain_margins = torch.cat(multi_domain_margins, dim=0)
                multi_domain_weights = torch.cat(multi_domain_weights, dim=0)
                multi_domain_weights = multi_domain_weights / torch.sum(multi_domain_weights, dim=0)
                sum_margin = torch.sum(multi_domain_margins * multi_domain_weights, dim=0)
                total_margins.append(sum_margin)

            if torch.sum(unknown_domain_mask) != 0:
                unknown_domain_feats = feats[unknown_domain_mask]
                app_pred = torch.zeros_like(out[unknown_domain_mask])
                unknown_weights = []
                unknown_logists = []

                for i in range(self.n_multi_classfiers):
                    ds_centroids = F.normalize(centroids[i], dim=1)
                    ds_norm_feats = F.normalize(unknown_domain_feats, dim=1)
                    ds_logits = self.multi_classifier(unknown_domain_feats, i)
                    sims = torch.mm(ds_norm_feats, ds_centroids.T) + 0.00001

                    target_sim_thrs = self.center_sim_thrs[i]
                    target_sim_thrs = target_sim_thrs.repeat((len(sims), 1)).cuda()
                    thr_mask = sims > target_sim_thrs
                    tmp_probs = abs((1 - (1 - sims) / (1 - target_sim_thrs + 0.00001)) * thr_mask) + 0.00001
                    prob_sims = torch.max(tmp_probs, dim=1).values

                    # print(sims.shape, thr_mask.shape, tmp_probs.shape)
                    # print(prob_sims.shape)

                    # prob_sims = torch.masked_scatter(prob_sims, thr_mask, sims[thr_mask])
                    # print(i, prob_sims)
                    # prob_sims = torch.max(tmp_probs, dim=1).values
                    # print(prob_sims.shape)
                    prob_sims = prob_sims.unsqueeze(1).repeat((1, sims.shape[1]))
                    unknown_weights.append(prob_sims.unsqueeze(0))
                    unknown_logists.append(ds_logits.unsqueeze(0))

                unknown_weights = torch.cat(unknown_weights, dim=0)
                unknown_logists = torch.cat(unknown_logists, dim=0)

                unknown_weights = unknown_weights / torch.sum(unknown_weights, dim=0)
                app_pred = torch.sum(unknown_weights * unknown_logists, dim=0)
                probs = F.softmax(app_pred, dim=1)
                probs = torch.topk(probs, k=2, dim=1).values
                margin = probs[:, 0] - probs[:, 1]
                total_margins.append(margin)
        total_margins = torch.cat(total_margins, dim=-1)
        return total_margins



    def cal_domain_margin(self, data_loader, centroids):
        self.net.eval()
        self.multi_classifier.eval()
        total_margins = []
        for x, _, ind, ds_ind in data_loader:
            x, ds_ind = x.cuda(), ds_ind.cuda()

            with torch.no_grad():
                out, feats = self.net(x)

                known_domain_mask = (ds_ind < self.n_multi_classfiers)
                unknown_domain_mask = (ds_ind == self.n_multi_classfiers)

                if torch.sum(known_domain_mask) != 0:
                    multi_domain_preds = []
                    multi_domain_margins = []
                    multi_domain_weights = []
                    for i in range(self.n_multi_classfiers):
                        preds = self.multi_classifier(feats[known_domain_mask], i)
                        multi_domain_preds.append(preds)
                    for i in range(self.n_multi_classfiers):
                        ds_mask = ds_ind[known_domain_mask] == i
                        ds_weights = torch.zeros_like(ds_mask, dtype=torch.float)
                        ds_weights = torch.masked_fill(ds_weights, ds_mask, 1.)


                        ds_centroids = F.normalize(centroids[i], dim=1)
                        ds_norm_feats = F.normalize(feats[known_domain_mask][~ds_mask], dim=1)
                        sims = torch.mm(ds_norm_feats, ds_centroids.T)

                        target_sim_thrs = self.center_sim_thrs[i]
                        target_sim_thrs = target_sim_thrs.repeat((len(sims), 1)).cuda()
                        thr_mask = sims > target_sim_thrs
                        thr_mask = torch.sum(thr_mask, dim=1) > 0.
                        thr_mask = torch.tensor(thr_mask, dtype=torch.float)
                        # print('11',thr_mask)

                        ds_weights = torch.masked_scatter(ds_weights, ~ds_mask, thr_mask)
                        multi_domain_weights.append(ds_weights.unsqueeze(0))


                        probs = F.softmax(multi_domain_preds[i], dim=1)
                        probs = torch.topk(probs, k=2, dim=1).values
                        margin = probs[:, 0] - probs[:, 1]
                        multi_domain_margins.append(margin.unsqueeze(0))

                    multi_domain_margins = torch.cat(multi_domain_margins, dim=0)
                    multi_domain_weights = torch.cat(multi_domain_weights, dim=0)
                    multi_domain_weights = multi_domain_weights / torch.sum(multi_domain_weights, dim=0)
                    sum_margin = torch.sum(multi_domain_margins * multi_domain_weights, dim=0)
                    del multi_domain_margins, multi_domain_weights
                    total_margins.append(sum_margin)

                if torch.sum(unknown_domain_mask) != 0:
                    unknown_domain_feats = feats[unknown_domain_mask]
                    unknown_weights = []
                    unknown_logists = []

                    for i in range(self.n_multi_classfiers):
                        ds_centroids = F.normalize(centroids[i], dim=1)
                        ds_norm_feats = F.normalize(unknown_domain_feats, dim=1)
                        ds_logits = self.multi_classifier(unknown_domain_feats, i)
                        sims = torch.mm(ds_norm_feats, ds_centroids.T) + 0.00001

                        target_sim_thrs = self.center_sim_thrs[i]
                        target_sim_thrs = target_sim_thrs.repeat((len(sims), 1)).cuda()
                        thr_mask = sims > target_sim_thrs
                        tmp_probs = abs((1 - (1 - sims) / (1 - target_sim_thrs + 0.00001)) * thr_mask) + 0.00001
                        prob_sims = torch.max(tmp_probs, dim=1).values
                        prob_sims = prob_sims.unsqueeze(1).repeat((1, sims.shape[1]))
                        unknown_weights.append(prob_sims.unsqueeze(0))
                        unknown_logists.append(ds_logits.unsqueeze(0))

                    unknown_weights = torch.cat(unknown_weights, dim=0)
                    unknown_logists = torch.cat(unknown_logists, dim=0)

                    unknown_weights = unknown_weights / torch.sum(unknown_weights, dim=0)
                    app_pred = torch.sum(unknown_weights * unknown_logists, dim=0)
                    probs = F.softmax(app_pred, dim=1)
                    probs = torch.topk(probs, k=2, dim=1).values
                    margin = probs[:, 0] - probs[:, 1]
                    total_margins.append(margin)
        total_margins = torch.cat(total_margins, dim=-1)
        return total_margins



    def cal_domain_margin1(self, data_loader, centroids):
        self.net.eval()
        self.multi_classifier.eval()
        total_margins = []
        for x, y, ind, ds_ind in data_loader:
            x, ds_ind = x.cuda(), ds_ind.cuda()


            with torch.no_grad():
                out, feats = self.net(x)

                known_domain_mask = (ds_ind < self.n_multi_classfiers)
                unknown_domain_mask = (ds_ind == self.n_multi_classfiers)

                if torch.sum(known_domain_mask) != 0:
                    # multi_domain_preds = []
                    multi_domain_margins = []
                    multi_domain_weights = []
                    for i in range(self.n_multi_classfiers):
                        preds = self.multi_classifier(feats[known_domain_mask], i)
                        _, preds_cls = torch.max(preds, 1)

                        ds_mask = ds_ind[known_domain_mask] == i
                        ds_weights = torch.zeros_like(ds_mask, dtype=torch.float)
                        ds_weights = torch.masked_fill(ds_weights, ds_mask, 1.)


                        ds_centroids = F.normalize(centroids[i], dim=1)

                        ds_pred_feats = ds_centroids[preds_cls]

                        ds_norm_feats = F.normalize(feats[known_domain_mask][~ds_mask], dim=1)
                        sims = torch.mm(ds_norm_feats, ds_pred_feats.T)
                        sims = torch.diag(sims)

                        target_sim_thrs = self.center_sim_thrs[i][preds_cls[~ds_mask]]
                        thr_mask = sims > target_sim_thrs
                        thr_mask = torch.tensor(thr_mask, dtype=torch.float)

                        ds_weights = torch.masked_scatter(ds_weights, ~ds_mask, thr_mask)
                        multi_domain_weights.append(ds_weights.unsqueeze(0))


                        probs = F.softmax(preds, dim=1)
                        probs = torch.topk(probs, k=2, dim=1).values
                        margin = probs[:, 0] - probs[:, 1]
                        multi_domain_margins.append(margin.unsqueeze(0))

                    multi_domain_margins = torch.cat(multi_domain_margins, dim=0)
                    multi_domain_weights = torch.cat(multi_domain_weights, dim=0)
                    multi_domain_weights = multi_domain_weights / torch.sum(multi_domain_weights, dim=0)
                    # print(torch.sum(multi_domain_weights != 1.))
                    sum_margin = torch.sum(multi_domain_margins * multi_domain_weights, dim=0)
                    del multi_domain_margins, multi_domain_weights
                    total_margins.append(sum_margin)

                if torch.sum(unknown_domain_mask) != 0:
                    unknown_domain_feats = feats[unknown_domain_mask]
                    unknown_weights = []
                    unknown_logists = []

                    for i in range(self.n_multi_classfiers):
                        ds_centroids = F.normalize(centroids[i], dim=1)
                        ds_norm_feats = F.normalize(unknown_domain_feats, dim=1)
                        ds_logits = self.multi_classifier(unknown_domain_feats, i)
                        _, preds_cls = torch.max(ds_logits, 1)

                        sims = (torch.mm(ds_norm_feats, ds_centroids[preds_cls].T) + 0.00001)
                        sims = torch.diag(sims)

                        target_sim_thrs = self.center_sim_thrs[i][preds_cls].cuda()
                        # target_sim_thrs = target_sim_thrs.repeat((len(sims), 1)).cuda()
                        thr_mask = sims > target_sim_thrs
                        tmp_probs = abs((1 - (1 - sims) / (1 - target_sim_thrs + 0.00001)) * thr_mask) + 0.00001

                        # prob_sims = torch.max(tmp_probs, dim=1).values
                        tmp_probs = tmp_probs.unsqueeze(1).repeat((1, ds_logits.shape[1]))
                        # prob_sims = prob_sims.unsqueeze(1).repeat((1, sims.shape[1]))
                        unknown_weights.append(tmp_probs.unsqueeze(0))
                        unknown_logists.append(ds_logits.unsqueeze(0))

                    unknown_weights = torch.cat(unknown_weights, dim=0)
                    unknown_logists = torch.cat(unknown_logists, dim=0)

                    unknown_weights = unknown_weights / torch.sum(unknown_weights, dim=0)
                    # print(torch.sum(unknown_weights != 1.))

                    app_pred = torch.sum(unknown_weights * unknown_logists, dim=0)
                    # exit()
                    probs = F.softmax(app_pred, dim=1)
                    probs = torch.topk(probs, k=2, dim=1).values
                    margin = probs[:, 0] - probs[:, 1]
                    total_margins.append(margin)
        total_margins = torch.cat(total_margins, dim=-1)
        return total_margins




    def predict_multi(self, data_loader, centroids):
        self.net.eval()
        self.multi_classifier.eval()

        # probs = []
        logits = []
        # embedding_features = []


        for x, y, ind, ds_ind in data_loader:
            x, y, ds_ind = x.cuda(), y.cuda(), ds_ind.cuda()

            with torch.no_grad():
                out, feats = self.net(x)
            logit = torch.zeros_like(out).cuda()

            for i in range(self.n_multi_classfiers):
                ds_mask = ds_ind == i
                pred = self.multi_classifier(feats[ds_mask], i)
                logit = torch.masked_scatter(logit, ds_mask.unsqueeze(1).repeat((1, self.net.n_label)), pred)

            unknown_domain_mask = (ds_ind == self.n_multi_classfiers)
            if torch.sum(unknown_domain_mask) != 0:
                unknown_domain_feats = feats[unknown_domain_mask]
                app_pred = torch.zeros_like(out[unknown_domain_mask])
                # print('unknown domain', app_pred.shape)
                unknown_weights = []
                unknown_logists = []


                for i in range(self.n_multi_classfiers):
                    ds_centroids = F.normalize(centroids[i], dim=1)
                    ds_norm_feats = F.normalize(unknown_domain_feats, dim=1)
                    ds_logits = self.multi_classifier(unknown_domain_feats, i)
                    sims = torch.mm(ds_norm_feats, ds_centroids.T) + 0.00001

                    unknown_weights.append(sims.unsqueeze(0))
                    unknown_logists.append(ds_logits.unsqueeze(0))
                    # print(sims)
                    # print(sims.shape, ds_logits.shape)
                    # weight = sims / torch.sum(sims)
                    # ds_weighted_pred = ds_logits * sims
                    # app_pred += ds_weighted_pred
                unknown_weights = torch.cat(unknown_weights, dim=0)
                unknown_logists = torch.cat(unknown_logists, dim=0)

                unknown_weights = unknown_weights / torch.sum(unknown_weights, dim=0)


                # print(unknown_weights[0], torch.max(unknown_weights[0], dim=-1).values)
                # print(unknown_weights[1], torch.max(unknown_weights[1], dim=-1).values)
                # print(unknown_weights[2])
                # exit()

                app_pred = torch.mean(unknown_weights * unknown_logists, dim=0)
                logit = torch.masked_scatter(logit, unknown_domain_mask.unsqueeze(1).repeat((1, self.net.n_label)), app_pred)
            logits.append(logit)

        logits = torch.cat(logits, dim=0).cpu()

        probs = F.softmax(logits, dim=1)

        return probs

    def predict_multi_new(self, data_loader, centroids):
        self.net.eval()
        self.multi_classifier.eval()

        # probs = []
        logits = []
        # embedding_features = []

        ds_sim_thr = self.get_ds_sim_thr()

        for x, y, ind, ds_ind in data_loader:
            x, y, ds_ind = x.cuda(), y.cuda(), ds_ind.cuda()

            with torch.no_grad():
                out, feats = self.net(x)
            # logit = torch.zeros_like(out).cuda()
            logit = torch.zeros((self.n_multi_classfiers, out.shape[0], out.shape[1]))
            # print(ds_ind)
            # print(self.n_multi_classfiers)

            for i in range(self.n_multi_classfiers):
                pred = self.multi_classifier(feats, i)
                # logit = torch.masked_scatter(logit, ds_mask.unsqueeze(1).repeat((1, self.net.n_label)), pred)
                logit[i] = pred

            unknown_domain_mask = (ds_ind == self.n_multi_classfiers)
            if torch.sum(unknown_domain_mask) != 0:
                unknown_domain_feats = feats[unknown_domain_mask]
                app_pred = torch.zeros_like(out[unknown_domain_mask])
                # print('unknown domain', app_pred.shape)
                unknown_weights = []
                unknown_logists = []

                for i in range(self.n_multi_classfiers):
                    ds_centroids = F.normalize(centroids[i], dim=1)
                    ds_norm_feats = F.normalize(unknown_domain_feats, dim=1)
                    ds_logits = self.multi_classifier(unknown_domain_feats, i)
                    sims = torch.mm(ds_norm_feats, ds_centroids.T) + 0.00001

                    unknown_weights.append(sims.unsqueeze(0))
                    unknown_logists.append(ds_logits.unsqueeze(0))
                    # print(sims)
                    # print(sims.shape, ds_logits.shape)
                    # weight = sims / torch.sum(sims)
                    # ds_weighted_pred = ds_logits * sims
                    # app_pred += ds_weighted_pred
                unknown_weights = torch.cat(unknown_weights, dim=0)
                unknown_logists = torch.cat(unknown_logists, dim=0)

                unknown_weights = unknown_weights / torch.sum(unknown_weights, dim=0)

                # print(unknown_weights[0], torch.max(unknown_weights[0], dim=-1).values)
                # print(unknown_weights[1], torch.max(unknown_weights[1], dim=-1).values)
                # print(unknown_weights[2])
                # exit()

                app_pred = torch.mean(unknown_weights * unknown_logists, dim=0)
                logit = torch.masked_scatter(logit, unknown_domain_mask.unsqueeze(1).repeat((1, self.net.n_label)),
                                             app_pred)
            logits.append(logit)

        logits = torch.cat(logits, dim=0).cpu()

        probs = F.softmax(logits, dim=1)

        return probs

    def get_ds_sim_thr(self):
        sim_thr = torch.zeros((self.n_multi_classfiers, self.net.n_label))
        label_loader = self.build_ds_sequence_train_label_loader()
        l_embedding, l_true_labels, l_ds_inds = self.predict_embed(label_loader)
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
                    sim_thr[i][j] = torch.max(cos_sim)
        return sim_thr

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






