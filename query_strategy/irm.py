from .base_strategy import BaseStrategy
import numpy as np


class IRMSampling(BaseStrategy):

    def query(self, n_select):
        _, unlabel_ind = self.build_unlabel_loader()
        tmp_ind = np.random.permutation(range(len(unlabel_ind)))[:n_select]
        # center_feature = self.trainer.center_loss.centers



        return unlabel_ind[tmp_ind]