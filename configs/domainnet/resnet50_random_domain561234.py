_base_ = [
    'base_resnet50_domain561234.py'
]

strategy_params=dict(
    type='RandomSampling'
)