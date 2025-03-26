_base_ = [
    'base_resnet50_domain_best.py'
]

strategy_params=dict(
    type='CLUESampling',
    # type='RandomSampling'

)


