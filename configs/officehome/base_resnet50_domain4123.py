model = dict(type='resnet50', n_classes=65, pretained=True)

dataset = dict(type='OfficeHome', root='./data/officehome',
               ann_files = dict(
                   train_anns=['domain4_train.txt', 'domain1_train.txt', 'domain2_train.txt', 'domain3_train.txt'],
                   test_anns=['domain4_test.txt', 'domain1_test.txt', 'domain2_test.txt', 'domain3_test.txt']),
               kwargs=dict())

active_cfg = dict(n_round=7, n_initial=1000, n_select=500, select_bs=256, sub_num=1024)
train_cfg = dict(type='BaseTrainer', epochs=50, val_interval=25, train_bs=128, test_bs=256,
                 # optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
                 # scheduler=dict(type='MultiStepLR', milestones=[60, 80], gamma=0.1),
                 optimizer=dict(type='Adadelta', lr=1.0),
                 scheduler=dict(type='None')
                 )
label_info = dict(label_ind=[], unlabel_ind=[])

