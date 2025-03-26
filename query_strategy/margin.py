from .base_strategy import BaseStrategy
import numpy as np
import torch

class MarginSampling(BaseStrategy):

    def query(self, n_select):
        print('start margin')

        # self.multi_classifier = self.trainer.multi_classifier
        # self.net = self.trainer.net
        self.net = self.net[0]


        unlabel_loader, unlabel_ind = self.build_unlabel_loader()
        # tmp_ind = np.random.permutation(range(len(unlabel_ind)))[:n_select]

        predicts, _, _, _ = self.predict_probs_and_embed(unlabel_loader)
        pros = torch.topk(predicts,k=2, dim=1).values
        delta = pros[:,0] - pros[:,1]
        entropy_ind = torch.argsort(delta, descending=True)[-n_select:].cpu()

        return unlabel_ind[entropy_ind]