_base_ = [
    'base_resnet50_domain3412.py'
]

strategy_params=dict(
    type='CoresetSampling',
)