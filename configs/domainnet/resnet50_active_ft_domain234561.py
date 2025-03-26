_base_ = [
    'base_resnet50_domain234561.py'
]

strategy_params=dict(
    type='ActiveFTSampling',
    lr=0.001,
    max_iter=100,
    batch_size=10000,
    temperature=0.07
)