from .base_trainer import BaseTrainer
import torch
from tqdm import tqdm
from trainer.losses import CenterSeperateMarginLoss
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import time
from torch import nn
from .losses import *

class STDTrainer(BaseTrainer):
    def __init__(self, net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg=None, is_amp=False, logger=None):
        super().__init__(net, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg, is_amp, logger)
        self.net = net[0]

