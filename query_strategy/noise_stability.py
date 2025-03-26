import torch

from .base_strategy import BaseStrategy
import numpy as np
from sklearn.metrics import pairwise_distances
from utils import *
from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear
import torch.nn.functional as F
import copy
from tqdm import tqdm

class NoisySampling(BaseStrategy):

    def query(self, n):
        self.net = self.net[0]
        unlabel_loader, unlabel_ind = self.build_unlabel_loader()

        # uncertainty = torch.zeros(SUBSET).cpu()

        diffs = torch.tensor([]).cpu()
        use_feature = False
        outputs = self.get_all_outputs(self.net, unlabel_loader, use_feature)
        for i in range(5):
            print('unlabel %d/%d' % (i+1,5))
            noisy_model = copy.deepcopy(self.net)
            noisy_model.eval()

            noisy_model.apply(self.add_noise_to_weights)
            outputs_noisy = self.get_all_outputs(noisy_model, unlabel_loader, use_feature)

            diff_k = outputs_noisy - outputs
            for j in range(diff_k.shape[0]):
                diff_k[j, :] /= outputs[j].norm()
            diffs = torch.cat((diffs.cpu(), diff_k.cpu()), dim=1)

        indsAll, _ = self.kcenter_greedy(diffs, n)
        return unlabel_ind[indsAll]


    # def noise_stability_sampling(self, models, unlabeled_loader, n, args):
    #     # if NOISE_SCALE < 1e-8:
    #     #     uncertainty = torch.randn(SUBSET)
    #     #     return uncertainty
    #
    #     uncertainty = torch.zeros(SUBSET).cpu()
    #
    #     diffs = torch.tensor([]).cpu()
    #     use_feature = args.dataset in ['house']
    #     outputs = self.get_all_outputs(models['backbone'], unlabeled_loader, use_feature)
    #     for i in range(args.n_sampling):
    #         noisy_model = copy.deepcopy(models['backbone'])
    #         noisy_model.eval()
    #
    #         noisy_model.apply(self.add_noise_to_weights)
    #         outputs_noisy = self.get_all_outputs(noisy_model, unlabeled_loader, use_feature)
    #
    #         diff_k = outputs_noisy - outputs
    #         for j in range(diff_k.shape[0]):
    #             diff_k[j, :] /= outputs[j].norm()
    #         diffs = torch.cat((diffs, diff_k), dim=1)
    #
    #     indsAll, _ = self.kcenter_greedy(diffs, n)
    #     return unlabel_ind[entropy_ind]
    #
    #     # print(indsAll)
    #     for ind in indsAll:
    #         uncertainty[ind] = 1
    #
    #     return uncertainty.cpu()

    def add_noise_to_weights(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):  # and isinstance(m, nn.Conv2d):
                noise = torch.randn(m.weight.size())
                noise = noise.cuda()
                noise *= (self.strategy_cfg.NOISE_SCALE * m.weight.norm() / noise.norm())
                m.weight.add_(noise)

    def k_dpp(self, X, K):
        DPP = FiniteDPP('likelihood', **{'L_gram_factor': 1e6 * X.cpu().numpy().transpose()})
        DPP.flush_samples()
        DPP.sample_mcmc_k_dpp(size=K)
        indsAll = DPP.list_of_samples[0][0]
        return indsAll

    def kcenter_greedy(self, X, K):
        if K <= 0:
            return list(), list()
        elif K >= X.shape[0]:
            return list(range(X.shape[0])), list(range(X.shape[0]))

        # avg_norm = np.mean([torch.norm(X[i]).item() for i in range(X.shape[0])])
        mu = torch.zeros(1, X.shape[1]).cpu()
        D2 = torch.norm(X, dim=1)
        nearestId = -torch.ones(X.shape[0], dtype=torch.long).cpu()
        indsAll = []
        while len(indsAll) < K:
            for i, ind in enumerate(D2.topk(1)[1]):
                # if i == 0:
                #     print(len(indsAll), ind.item(), D2[ind].item(), X[ind,:5])
                D2[ind] = 0
                nearestId[ind] = ind
                mu = torch.cat((mu, X[ind].unsqueeze(0)), 0)
                indsAll.append(ind)

            newD = torch.cdist(X, mu[-1:]).squeeze(1)
            less_D2_mask = (D2 > newD)
            D2[less_D2_mask] = newD[less_D2_mask]
            nearestId[less_D2_mask] = indsAll[-1]

        # selected_norm = np.mean([torch.norm(X[i]).item() for i in indsAll])

        return torch.tensor(indsAll), nearestId

    def get_all_outputs(self, model, unlabeled_loader, use_feature=False):
        model.eval()
        outputs = torch.tensor([]).cuda()
        with torch.no_grad():
            for inputs, y, ind, _ in tqdm(unlabeled_loader):
            # for inputs, _, _ in unlabeled_loader:
                inputs = inputs.cuda()
                out, fea = model(inputs)
                if use_feature:
                    out = fea.cuda()
                else:
                    out = F.softmax(out, dim=1).cuda()
                outputs = torch.cat((outputs, out), dim=0)

        return outputs