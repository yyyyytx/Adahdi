_base_ = [
    'base_resnet50_domain123456.py'
]

dataset = dict(type='6FoldDomainnet')


strategy_params=dict(
    type='TQSSampling',
)

multi_classify = dict(
    type='MultiClassify',
    bottleneck_dim=2048,
    class_num=_base_.model.n_classes
)

discriminator = dict(
    type='Discriminator',
    bottleneck_dim=2048,
)

train_cfg = dict(type='TQSTrainer')