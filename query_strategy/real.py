import torch

from .base_strategy import BaseStrategy
import numpy as np
from sklearn.metrics import pairwise_distances
from utils import *
from sklearn.cluster import MiniBatchKMeans
import faiss
from collections import Counter
import random


def kmeanspp(ncentroids, feat):
    """
    K-means++
    Args:
      ncentroids (int):
      feat: [n, dim]
    """
    dim = feat.shape[-1]
    kmeans = MiniBatchKMeans(n_clusters = ncentroids, random_state=0, n_init=3, max_iter=100)# default is k-means++
    kmeans.fit(feat)
    index = faiss.IndexFlatL2(dim)
    index.add(kmeans.cluster_centers_)
    D, I = index.search(feat, 1)
    I = I.flatten() # list of cluster assignment for all unlabeled ins
    return  I

def compute_entropy( P ):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nansum(-P * np.log(P), axis=1)

class RealSampling(BaseStrategy):
    '''
    ECML/PKDD 2023 Paper: REAL: A Representative Error-Driven Approach for Active Learning
    '''

    def query(self, n_sample):
        self.net = self.net[0]


        # label_loader, label_ind = self.build_labeled_loader()
        unlabel_loader, unlabel_ind = self.build_unlabel_loader()
        unlabeled_pred, unlabeled_feat, _, _ = self.predict_probs_and_embed(unlabel_loader)

        unlabeled_pred = unlabeled_pred.numpy()
        unlabeled_feat = unlabeled_feat.numpy()
        ncentroids = self.net.n_label
        N = unlabeled_pred.shape[0]
        unlabeled_pseudo = np.argmax(unlabeled_pred, axis=-1)
        entropy = compute_entropy(unlabeled_pred)
        sample_idx, save_idx = [], []

        I = kmeanspp(ncentroids, unlabeled_feat)

        clu_value = [0] * ncentroids  # cluster value, more error, more valuable
        clu_majlbl = [-1] * ncentroids  # cluster value, more error, more valuable
        dis = np.zeros(N)
        # pass 1: fill clu_value
        for i in range(ncentroids):
            clu_sel = (I == i)  # selector for current cluster
            print(clu_sel)
            if np.sum(clu_sel) == 0: continue  # ，faiss
            cnt = Counter()
            for z in unlabeled_pseudo[clu_sel]:
                cnt[z] += 1
            # select minority from cnt
            lbl_freq = list(cnt.items())
            lbl_freq.sort(key=lambda x: x[1])
            clu_pseudo = unlabeled_pseudo[clu_sel]
            majlbl = lbl_freq[-1][0]  # the majority label
            clu_majlbl[i] = majlbl
            majscore = unlabeled_pred[clu_sel][:, majlbl]
            dismaj = 1 - majscore
            dis[clu_sel] = dismaj
            nonmaj_sel = clu_pseudo != majlbl
            # clu_value[i] = np.mean(dismaj[nonmaj_sel]) # set i，
            clu_value[i] = np.sum(dismaj[nonmaj_sel])

        # pass 2: sample proportionanlly to clu_value
        cvsm = np.sum(clu_value)
        clu_nsample = [int(i / cvsm * n_sample) for i in clu_value]
        nmissing = n_sample - np.sum(clu_nsample)
        highclui = np.argsort(clu_value)[::-1][:nmissing]
        for i in highclui:
            clu_nsample[i] += 1
        print(clu_nsample)
        print(torch.sum(torch.tensor(clu_nsample)))

        for i in range(ncentroids):
            clu_sel = (I == i)  # selector for current cluster
            topk = clu_nsample[i]  # TODO: topk > clu_size, qqp
            if topk <= 0: continue
            majlbl = clu_majlbl[i]
            clu_pseudo = unlabeled_pseudo[clu_sel]
            majscore = unlabeled_pred[clu_sel][:, majlbl]
            nonmaj_sel = clu_pseudo != majlbl
            nonmaj_idx = np.arange(len(clu_pseudo))[nonmaj_sel]  #
            # npseudoerr = np.sum(nonmaj_sel)
            npseudoerr = torch.sum(torch.tensor(nonmaj_sel))
            if npseudoerr > topk:  # topk
                # random or entropy pick topk from w
                picki = random.sample(nonmaj_idx.tolist(), topk)
            else:
                picki = np.argsort(majscore)[:topk]
            tmp = np.arange(len(I))[clu_sel][picki]
            sample_idx += tmp.tolist()

        dis_rank = np.argsort(dis)[::-1]  # big
        i = 0
        # ，
        tmp_idx = []
        labeled = [False] * N
        for i in sample_idx:
            labeled[i] = True
            tmp_idx.append(i)
        while len(tmp_idx) < n_sample:
            j = dis_rank[i]
            if not labeled[j]:
                sample_idx.append(j)
                labeled[j] = True
                tmp_idx.append(j)
            i += 1

        print(len(tmp_idx))
        assert len(tmp_idx) == n_sample
        tmp_idx = np.array(tmp_idx)

        return unlabel_ind[tmp_idx]


