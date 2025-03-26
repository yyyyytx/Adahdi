_base_ = [
    'base_resnet50_domain612345.py'
]

strategy_params=dict(
    type='RandomSampling'
)