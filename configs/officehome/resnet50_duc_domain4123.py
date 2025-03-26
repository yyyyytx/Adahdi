_base_ = [
    'base_resnet50_domain4123.py'
]

strategy_params=dict(
    type='DUCSampling',
)

train_cfg = dict(type='DUCTrainer', train_bs=64,optimizer=dict(type='Adadelta', lr=1.0))


