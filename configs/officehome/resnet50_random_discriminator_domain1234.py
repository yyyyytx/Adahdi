_base_ = [
    'base_resnet50_domain1234.py'
]

strategy_params=dict(
    type='RandomSampling'
)

multi_classifier=dict(
    type='MultiClassifier',
    embedding_size=2048,
    class_num=_base_.model.n_classes,
    n_classifier=1
)

multi_discriminator=dict(
    type='MultiDiscriminator',
    embedding_size=2048,
    num_domains=1,
)

domain_discriminator=dict(
    type='DomainDiscriminator',
    input_dim=2048,
    hidden_dim=2048,
    num_domains=1,

)

train_cfg = dict(type='DiscriminatorMultiClassifierTrainer',alpha=0.1)