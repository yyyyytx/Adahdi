_base_ = [
    'base_resnet50_domain345612.py'
]

strategy_params=dict(
    type='AlphaMixSampling',
    alpha_closed_form_approx=True,
    alpha_cap=0.2,
    alpha_opt=True
)