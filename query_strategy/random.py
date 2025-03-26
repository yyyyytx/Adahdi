from .base_strategy import BaseStrategy
import numpy as np


class RandomSampling(BaseStrategy):

    def query(self, n_select):
        _, unlabel_ind = self.build_unlabel_loader()
        tmp_ind = np.random.permutation(range(len(unlabel_ind)))[:n_select]
        return unlabel_ind[tmp_ind]