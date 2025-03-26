_base_ = [
    'base_resnet50_domain234561.py'
]

strategy_params=dict(
    type='NoiseSampling',
    K=10,
    NOISE_SCALE=0.001

)