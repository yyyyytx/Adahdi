_base_ = [
    'base_resnet50_domain1234.py'
]

strategy_params=dict(
    type='RandomSampling'
)

