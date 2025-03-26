_base_ = [
    'base_resnet50_domain123456.py'
]

strategy_params=dict(
    type='CLUESampling',
    # type='RandomSampling'

)


