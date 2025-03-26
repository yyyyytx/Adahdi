_base_ = [
    'base_resnet50_domain234561.py'
]

strategy_params=dict(
    type='CLUESampling',
    # type='RandomSampling'

)


