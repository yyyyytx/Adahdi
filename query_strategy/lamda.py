from .base_strategy import BaseStrategy
import numpy as np


class LAMDASampling(BaseStrategy):
    '''
    (ECCV2022)Combating Label Distribution Shift for Active Domain Adaptation
    '''

    def query(self, n_select):
        raise NotImplementedError