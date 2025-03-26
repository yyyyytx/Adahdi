model = dict(type='resnet50', n_classes=345, pretained=True)

dataset = dict(type='DomainNet', root='./data/DomainNet',
               ann_files = dict(
                   train_anns=['sketch_train.txt', 'clipart_train.txt', 'infograph_train.txt', 'painting_train.txt', 'quickdraw_train.txt', 'real_train.txt'],
                   test_anns=['sketch_test.txt', 'clipart_test.txt', 'infograph_test.txt', 'painting_test.txt', 'quickdraw_test.txt', 'real_test.txt'])
               )

active_cfg = dict(n_round=6, n_initial=2000, n_select=2000, select_bs=128, sub_num=512)
train_cfg = dict(type='BaseTrainer', epochs=50, val_interval=25, train_bs=128, test_bs=1024,
                 # optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
                 # scheduler=dict(type='MultiStepLR', milestones=[30], gamma=0.1)
                 optimizer=dict(type='Adadelta', lr=1.0),
                 scheduler=dict(type='None')
                 )
label_info = dict(label_ind=[], unlabel_ind=[])
