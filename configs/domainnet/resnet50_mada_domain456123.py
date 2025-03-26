_base_ = [
    'base_resnet50_domain456123.py'
]

model = dict(mada_module=True)

strategy_params=dict(
    type='MADASampling',
    LAMBDA_1=7,
    LAMBDA_2=0.5
)

train_cfg = dict(type='MADATrainer', BETA=1.0, LAMBDA=0.05, train_bs=32, optimizer=dict(type='Adadelta', lr=0.25))
