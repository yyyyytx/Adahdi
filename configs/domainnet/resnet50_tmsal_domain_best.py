_base_ = [
    'base_resnet50_domain_best.py'
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

train_cfg = dict(type='TMSALTrainer', epochs=50,
                 is_center=True, center_epoch=20, center_weights=[0., 1.], center_milestones=[0 ,20],
                 is_multi_classifier=True, multi_epochs=50, center_lr=0.1,multi_lr=0.01,
                 is_recall=True, is_margin=True, val_interval=50, alpha=0.8)