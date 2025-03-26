from .base_strategy import BaseStrategy
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans
import copy
import time

class VeSSALSampling(BaseStrategy):
    '''
    (ICML2023)Streaming Active Learning with Deep Neural Networks
    '''
    def query(self, n_select):
        self.net = self.net[0]
        self.zeta = 1
        label_loader, label_ind = self.build_labeled_loader()
        unlabel_loader, unlabel_ind = self.build_unlabel_loader()

        _, gradEmbedding, _, _ = self.predict_probs_and_embed(unlabel_loader)
        gradEmbedding = gradEmbedding.cpu().detach().numpy()
        cov_inv_scaling = 100
        start_time = time.time()
        chosen, skipped = self.streaming_sampler(gradEmbedding, n_select, early_stop=False, cov_inv_scaling=cov_inv_scaling)
        print('compute time (sec):', time.time() - start_time, flush=True)
        print('chosen: {}, skipped: {}, n:{}'.format(len(chosen), len(skipped), n_select), flush=True)
        if len(chosen) > n_select:
            chosen = chosen[:n_select]
        if len(chosen) < n_select:
            remain_ind = np.setdiff1d(np.arange(unlabel_ind), chosen)
            remain_ind = np.random.permutation(remain_ind)
            chosen = np.append(chosen, remain_ind[:n_select-len(chosen)])
        return unlabel_ind[chosen]


    def streaming_sampler(self, samps, k, early_stop=False, streaming_method='det', cov_inv_scaling=100):
        inds = []
        skipped_inds = []
        samps = samps.reshape((samps.shape[0], 1, samps.shape[1]))
        dim = samps.shape[-1]
        rank = samps.shape[-2]

        covariance = torch.zeros(dim, dim).cuda()
        covariance_inv = cov_inv_scaling * torch.eye(dim).cuda()
        samps = torch.tensor(samps)
        samps = samps.cuda()

        for i, u in enumerate(samps):
            if i % 1000 == 0: print(i, len(inds), flush=True)
            if rank > 1:
                u = torch.Tensor(u).t().cuda()
            else:
                u = u.view(-1, 1)

            # get determinantal contribution (matrix determinant lemma)
            if rank > 1:
                norm = torch.abs(torch.det(u.t() @ covariance_inv @ u))
            else:
                norm = torch.abs(u.t() @ covariance_inv @ u)

            ideal_rate = (k - len(inds)) / (len(samps) - (i))
            # just average everything together: \Sigma_t = (t-1)/t * A\{t-1}  + 1/t * x_t x_t^T
            covariance = (i / (i + 1)) * covariance + (1 / (i + 1)) * (u @ u.t())

            self.zeta = (ideal_rate / (torch.trace(covariance @ covariance_inv))).item()

            pu = np.abs(self.zeta) * norm

            if np.random.rand() < pu.item():
                inds.append(i)
                if early_stop and len(inds) >= k:
                    break

                # woodbury update to covariance_inv
                inner_inv = torch.inverse(torch.eye(rank).cuda() + u.t() @ covariance_inv @ u)
                inner_inv = self.inf_replace(inner_inv)
                covariance_inv = covariance_inv - covariance_inv @ u @ inner_inv @ u.t() @ covariance_inv
            else:
                skipped_inds.append(i)

        return inds, skipped_inds

    def inf_replace(self, mat):
        mat[torch.where(torch.isinf(mat))] = torch.sign(mat[torch.where(torch.isinf(mat))]) * np.finfo('float32').max
        return mat