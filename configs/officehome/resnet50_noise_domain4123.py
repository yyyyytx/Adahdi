_base_ = [
    'base_resnet50_domain4123.py'
]

strategy_params=dict(
    type='NoiseSampling',
    K=10,
    NOISE_SCALE=0.001

)