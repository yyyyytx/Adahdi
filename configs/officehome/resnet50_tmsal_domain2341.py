_base_ = [
    'base_resnet50_domain2341.py'
]

strategy_params=dict(
    type='TMSSampling'
    # type='RandomSampling'
)

multi_classifier=dict(
    type='MultiClassifier',
    embedding_size=2048,
    class_num=_base_.model.n_classes,
    n_classifier=1
)


train_cfg = dict(type='MarginTMSALTrainer',
                 epochs=50,
                 is_center=True,
                 center_epoch=20,
                 center_weights=[0., 1.],
                 center_milestones=[0 ,20],
                 is_multi_classifier=True,
                 multi_epochs=50,
                 center_lr=0.1,
                 multi_lr=1.0,
                 multi_optimizer=dict(type='Adadelta', lr=1.0),
                 multi_scheduler=None,
                 is_recall=True,
                 is_margin=True,
                 val_interval=25,
                 alpha=1.0)

