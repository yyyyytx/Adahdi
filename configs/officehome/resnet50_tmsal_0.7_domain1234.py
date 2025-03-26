_base_ = [
    'base_resnet50_domain1234.py'
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

train_cfg = dict(type='TMSALTrainer', epochs=100,
                 is_center=True, center_epoch=40, center_weights=[0., 1.], center_milestones=[0 ,40],
                 is_multi_classifier=True, multi_epochs=100,
                 is_recall=True, is_margin=True, val_interval=20,
                 alpha=0.7)
