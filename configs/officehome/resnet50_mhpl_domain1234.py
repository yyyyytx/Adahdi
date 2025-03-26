_base_ = [
    'base_resnet50_domain1234.py'
]



strategy_params=dict(
    type='MHPLSampling',
    S_K=10,

)
active_cfg = dict(sub_num=10240)

# train_cfg = dict(epochs=1)
