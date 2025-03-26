_base_ = [
    'base_resnet50_domain2341.py'
]

strategy_params=dict(
    type='NoiseSampling',
    K=10,
    NOISE_SCALE=0.001

)