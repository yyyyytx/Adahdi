from .base_strategy import BaseStrategy
import numpy as np


class ADSSampling(BaseStrategy):
    '''
    (AAAI2021)Agreement-Discrepancy-Selection: Active Learning with Progressive Distribution Alignment
    '''

    def query(self, n_select):
        raise NotImplementedError