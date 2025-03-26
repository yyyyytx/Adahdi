_base_ = [
    'base_resnet50_domain345612.py'
]

strategy_params=dict(
    type='SDMSampling',
    SDM_LAMBDA=0.01,
    SDM_MARGIN=1.0
)


train_cfg = dict(type='SDMTrainer')