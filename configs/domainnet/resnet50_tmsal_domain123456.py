_base_ = [
    'base_resnet50_domain123456.py'
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

# train_cfg = dict(type='BaseTrainer', epochs=50, val_interval=60, train_bs=128, test_bs=1024,
#                  optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
#                  scheduler=dict(type='MultiStepLR', milestones=[30], gamma=0.1)
#                  )

train_cfg = dict(type='MarginTMSALTrainer', epochs=50,
                 is_center=True, center_epoch=20, center_weights=[0., 1.], center_milestones=[0 ,20],
                 is_multi_classifier=True, multi_epochs=50, center_lr=0.1,
                 is_recall=True, is_margin=True, val_interval=50, alpha=1.0,
                 multi_lr=1.0,
                 multi_optimizer=dict(type='Adadelta', lr=1.0),
                 scheduler=dict(type='None')
                 )