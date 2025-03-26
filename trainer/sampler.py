import torch
import itertools
from torch.utils.data.sampler import Sampler
class InfiniteSampler(Sampler):
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size

    def __iter__(self):
        yield from itertools.islice(self._infinite(), 0, None, 1) # Infinite iterator

    def _infinite(self):
        g = torch.Generator()
        while True:
            yield from torch.randperm(self.dataset_size, generator=g)