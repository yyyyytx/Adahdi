_base_ = [
    'base_resnet50_domain345612.py'
]

strategy_params=dict(
    type='GradNormSampling',
    is_subset=False
)