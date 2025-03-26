_base_ = [
    'base_resnet50_domain2341.py'
]

strategy_params=dict(
    type='RandomSampling'
)