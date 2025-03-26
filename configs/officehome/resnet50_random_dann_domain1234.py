_base_ = [
    'base_resnet50_domain1234.py'
]

strategy_params=dict(
    type='RandomSampling'
)

domain_discriminator=dict(
    type='DomainDiscriminator',
    input_dim=2048,
    hidden_dim=2048,
    num_domains=1,

)

train_cfg = dict(type='DANNTrainer',alpha=0.1)