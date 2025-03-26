_base_ = [
    'base_resnet50_domain4123.py'
]

strategy_params=dict(
    type='TiDALSampling',
)

pred_module = dict(
    type='TDNet',
    feature_sizes=[32, 16, 8, 4],
    num_channels=[256, 512, 1024, 2048],
    interm_dim=128,
    out_dim=_base_.model.n_classes
)

train_cfg = dict(type='TiDALTrainer', WEIGHT=1.0)

