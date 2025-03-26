_base_ = [
    'base_resnet50_domain4123.py'
]

strategy_params=dict(
    type='RandomSampling'
)