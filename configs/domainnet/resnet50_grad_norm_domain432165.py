_base_ = [
    'base_resnet50_domain432165.py'
]

strategy_params=dict(
    type='GradNormSampling',
    is_subset=False
)