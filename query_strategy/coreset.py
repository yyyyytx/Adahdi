from .base_strategy import BaseStrategy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli
from tqdm import tqdm
class CoresetSampling(BaseStrategy):

    def query(self, n_select):
        self.net = self.net[0]

        label_loader, label_ind = self.build_labeled_loader()
        unlabel_loader, unlabel_ind = self.build_unlabel_loader()

        _, label_embedding, _, _ = self.predict_probs_and_embed(label_loader)
        _, unlabel_embedding, _, _ = self.predict_probs_and_embed(unlabel_loader)

        # probs, embeddings = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        select_inds = self.greedy_k_center(label_embedding, unlabel_embedding, n_select)

        return unlabel_ind[select_inds]

    def greedy_k_center(self, labeled, unlabeled, amount):
        greedy_indices = []
        # print(unlabeled.shape)
        print('cal minimum distances between the labeled and unlabeled examples')
        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = torch.min(torch.cdist(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), dim=0).values

        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = torch.cdist(labeled[j:j + 100, :], unlabeled)
            else:
                dist = torch.cdist(labeled[j:, :], unlabeled)
            min_dist = torch.vstack((min_dist, torch.min(dist, dim=0).values.reshape((1, min_dist.shape[1]))))
            min_dist = torch.min(min_dist, dim=0).values
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # print(min_dist.shape)
        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = torch.argmax(min_dist)
        # print(farthest)
        greedy_indices.append(farthest)
        print('select samples')
        for i in range(amount - 1):
            if i % 100 == 0:
                print("At Point %d/%d" % (i, amount))
            dist = torch.cdist(unlabeled[greedy_indices[-1], :].reshape((1, unlabeled.shape[1])), unlabeled)
            min_dist = torch.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = torch.min(min_dist, dim=0).values
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = torch.argmax(min_dist)
            greedy_indices.append(farthest)
        greedy_indices = torch.tensor(greedy_indices).cpu()
        return np.array(greedy_indices, dtype=int)


