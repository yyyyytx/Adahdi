_base_ = [
    'base_resnet50_domain234561.py'
]

strategy_params=dict(
    type='LASSampling',
    S_K = 10,
    S_M = 10,
    S_PROP_ITER = 1,
    S_PROP_COEF = 1.0
)

active_cfg = dict(sub_num=10240)
