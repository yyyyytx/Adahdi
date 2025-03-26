_base_ = [
    'base_resnet50_domain456123.py'
]

strategy_params=dict(
    type='LLALSampling',
    is_subset=False
)

additional_module=dict(type='LossNet',
                       feature_sizes=[32, 16, 8, 4],
                       num_channels=[256, 512, 1024, 2048],
                       interm_dim=128)

train_cfg = dict(epoch_loss=30, type='LLALTrainer', weight=1.0, margin=1.0)