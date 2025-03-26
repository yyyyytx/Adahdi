from .base_strategy import BaseStrategy
import numpy as np
import torch

class EntropySampling(BaseStrategy):

    def query(self, n_select):
        self.net = self.net[0]

        unlabel_loader, unlabel_ind = self.build_unlabel_loader()
        # tmp_ind = np.random.permutation(range(len(unlabel_ind)))[:n_select]

        predicts, _, _, _ = self.predict_probs_and_embed(unlabel_loader)
        entropy = (- predicts * torch.log2(predicts)).sum(dim=1)
        entropy_ind = torch.argsort(entropy, descending=True)[:n_select].cpu()



        return unlabel_ind[entropy_ind]