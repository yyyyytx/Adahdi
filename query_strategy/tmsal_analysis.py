import copy

from .base_strategy import BaseStrategy
import numpy as np
import torch.nn.functional as F
import torch

class TMSAnalysisSampling(BaseStrategy):

    def query(self, n_select):
        self.multi_classifier = self.trainer.multi_classifier
        self.n_multi_classfiers = len(self.multi_classifier.multi_classifiers)
        self.net = self.trainer.net
        self.center_sim_thrs = self.trainer.cal_domain_sim_thr().cuda()
        centroids = self.trainer.center_loss.centers


        loader_list = self.build_divided_unlabel_loader(self.active_cfg.sub_num,  shuffle=True)
        for i, unlabel_loader in enumerate(loader_list):
            print('unlabel loader: %d/%d' % (i + 1, len(loader_list)))
            self.analysis_correct(unlabel_loader, centroids)

    def analysis_correct(self, data_loader, centroids):
        self.net.eval()
        self.multi_classifier.eval()
        total_margins = []
        for x, ys, ind, ds_ind in data_loader:
            x, ys, ds_ind = x.cuda(), ys.cuda(), ds_ind.cuda()

            with torch.no_grad():
                out, feats = self.net(x)


            for i in range(len(self.multi_classifier.multi_classifiers)):
                ds_train_mask = self.cal_margin_grad(feats, ds_ind, i, ys, out)
                print('true domain {}:{}'.format(i, torch.sum(ds_train_mask)))

            known_domain_mask = (ds_ind < self.n_multi_classfiers)
            unknown_domain_mask = (ds_ind == self.n_multi_classfiers)
            

            # for i in range(len(self.multi_classifier.multi_classifiers)):
            #     known_domain_mask = (ds_ind == i)
            #     print('pred domain {}:{}'.format(i,torch.sum(known_domain_mask).item()))
            #     ds_mask = ds_ind[known_domain_mask] == i
            #
            #
            #     if torch.sum(known_domain_mask) != 0:
            #         preds = self.multi_classifier(feats[known_domain_mask], i)
            #         _, preds_cls = torch.max(preds, 1)
            #         ds_centroids = F.normalize(centroids[i], dim=1)
            #         ds_pred_feats = ds_centroids[preds_cls]
            #
            #         ds_pred_feats = ds_centroids[preds_cls]
            #
            #         ds_norm_feats = F.normalize(feats[known_domain_mask][~ds_mask], dim=1)
            #         sims = torch.mm(ds_norm_feats, ds_pred_feats.T)
            #         sims = torch.diag(sims)
            #
            #         target_sim_thrs = self.center_sim_thrs[i][preds_cls[~ds_mask]]
            #         thr_mask = sims > target_sim_thrs
            #         thr_mask = torch.tensor(thr_mask, dtype=torch.float)

                    # ds_weights = torch.masked_scatter(ds_weights, ~ds_mask, thr_mask)

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
            mask = 1.0 * delta <= ds_delta_logits #/ self.train_cfg.multi_epochs
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






