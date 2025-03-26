from .base_trainer import BaseTrainer
# from .irm_trainer import IRMTrainer
from .metric_trainer import MetricTrainer
from .llal_trainer import LLALTrainer
from .sdm_trainer import SDMTrainer
from .tqs_trainer import TQSTrainer
from .eada_trainer import EADATrainer
from .multi_classifier_trainer import MultiClassifierTrainer
from .recall_multi_classifier_trainer import RecallMultiClassifierTrainer
from .multi_discriminator_trainer import DiscriminatorMultiClassifierTrainer
from .dann_trainer import DANNTrainer
from .dann_multi_classifiers_trainer import DANNMultiClassifierTrainer
from .margin_tmsal_trainer import MarginTMSALTrainer
from .margin_tmsal_analysis_trainer import MarginTMSALTrainer_A
from .tidal_trainer import TiDALTrainer
from .mme_trainer import MMETrainer
from .group_dro import GroupDROTrainer
from .ltc_msda import LtCMSDATrainer
from .std_trainer import STDTrainer
from .duc_trainer import DUCTrainer
from .mcc_trainer import MCCTrainer
from .mada_trainer import MADATrainer
# def get_trainer(type)