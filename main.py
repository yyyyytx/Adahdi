import torch
from torch.utils.tensorboard import SummaryWriter
from mmengine.config import Config
from datasets import *
import argparse
from query_strategy import *
from query_strategy.noise_stability import NoisySampling
from utils import *
from models import *
import csv
from trainer import *
import time
import logging

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

query_strategy = {
    'RandomSampling': RandomSampling,
    'AlphaMixSampling': AlphaMixSampling,
    'IRMSampling': IRMSampling,
    'CDALSampling': CDALSampling,
    'GCNSampling': GCNSampling,
    'CoresetSampling':CoresetSampling,
    'BadgeSampling':BadgeSampling,
    'MetricSampling':MetricSampling,
    'LLALSampling':LLALSampling,
    'GradNormSampling':GradNormSampling,
    'EntropySampling':EntropySampling,
    'SDMSampling':SDMSampling,
    'TQSSampling':TQSSampling,
    'CLUESampling':CLUESampling,
    'TMSSampling':TMSSampling,
    'MarginSampling':MarginSampling,
    'ActiveFTSampling':ActiveFTSampling,
    'TiDALSampling':TiDALSampling,
    'LASSampling':LASSampling,
    'MHPLSampling':MHPLSampling,
    'RealSampling':RealSampling,
    'DUCSampling':DUCSampling,
    'VeSSALSampling':VeSSALSampling,
    'NoiseSampling':NoisySampling,
    'MADASampling':MADASampling

}


def active_train(cfg):
    model_cfg = cfg.model
    train_cfg = cfg.train_cfg
    strategy_cfg = cfg.strategy_params
    active_cfg = cfg.active_cfg
    dataset_cfg = cfg.dataset
    label_info = cfg.label_info

    assert strategy_cfg.type in query_strategy.keys()

    # log
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    writer = SummaryWriter(log_dir=cfg.output_dir)
    result_file = open(cfg.output_dir + '.csv', 'w')
    result_writer = csv.writer(result_file, quoting=csv.QUOTE_ALL)

    def get_logger(logger_name):
        logging.basicConfig(level=logging.INFO,
                            filename=logger_name)
        logger = logging.getLogger()
        return logger
    logger = get_logger(os.path.join(cfg.output_dir, 'info.log'))

    # dataset
    if dataset_cfg.type=='6FoldOfficeHome' or dataset_cfg.type=='6FoldDomainnet':
        train_ds, select_ds, test_ds, multi_train_ds = get_datasets(dataset_cfg.root, dataset_cfg.ann_files, dataset_cfg.type)
        train_cfg.multi_train_ds = multi_train_ds

    else:
        train_ds, select_ds, test_ds = get_datasets(dataset_cfg.root, dataset_cfg.ann_files, dataset_cfg.type)
    # print('domain len:', train_ds.domain_lens[0], train_ds.domain_lens[1])
    check_dataset_euqal(dataset_cfg.type, train_ds, select_ds)
    if len(train_ds.domain_lens) == 1:
        total_train_ind = np.arange(len(train_ds))
    else:
        total_train_ind = np.arange(train_ds.domain_lens[0]+train_ds.domain_lens[1])
    if dataset_cfg.type=='CIFAR10' or dataset_cfg.type=='CIFAR100' or dataset_cfg.type=='TinyImagenet' or dataset_cfg.type=='TinyImagenet_strong' or dataset_cfg.type=='CIFAR100_strong':
        label_ind = equal_num_initial(train_ds.targets, active_cfg.n_initial, model_cfg.n_classes)
    else:
        label_ind = np.random.permutation(train_ds.domain_lens[0])[:active_cfg.n_initial]
    # print(statistic_category_num(train_ds.targets, label_ind, model_cfg.n_classes))
    unlabel_ind = np.setdiff1d(total_train_ind, label_ind)
    label_info.label_ind = label_ind
    label_info.unlabel_ind = unlabel_ind

    print('start active learning stragety:' + strategy_cfg.type)
    print('seed:', cfg.seed)
    acc = np.zeros(active_cfg.n_round + 1)


    for rd in range(1, active_cfg.n_round + 1):
        # set_seeds(cfg.seed)
        print(torch.cuda.is_available())
        active_cfg.rd = rd
        labeled_ind_path = os.path.join(cfg.output_dir, 'label_ind_rd_%d.pt'%rd)
        if os.path.exists(labeled_ind_path) and cfg.debug==False:
            print('load label ind at round %d' % rd)
            print('load path:', labeled_ind_path)
            label_ind = torch.load(labeled_ind_path).numpy()
            domain_lens = len(train_ds.domain_lens)
            if rd + 1 > domain_lens:
                total_train_ind = np.arange(len(train_ds))
            else:
                count = 0
                for i in range(rd + 1):
                    count += train_ds.domain_lens[i]
                total_train_ind = np.arange(count)
            unlabel_ind = np.setdiff1d(total_train_ind, label_ind)
            label_info.label_ind=label_ind
            label_info.unlabel_ind=unlabel_ind
            continue
        else:
            # train_ds_num = rd+1
            if rd > len(train_ds.domain_lens):
                train_ds_num = len(train_ds.domain_lens)
            else:
                train_ds_num = rd
            train_cfg.dataset_number=train_ds_num

            domain_lens = len(train_ds.domain_lens)
            if rd + 1 > domain_lens:
                total_train_ind = np.arange(len(train_ds))
            else:
                count = 0
                for i in range(rd + 1):
                    count += train_ds.domain_lens[i]
                total_train_ind = np.arange(count)

            label_info.l_train_ds_number = train_ds_num
            label_ind = label_info.label_ind
            unlabel_ind = np.setdiff1d(total_train_ind, label_ind)
            label_info.label_ind = label_ind
            label_info.unlabel_ind = unlabel_ind
            print('start training round %d' % rd)
            print('labeled data %d unlabeled data %d' % (len(label_info.label_ind), len(label_info.unlabel_ind)))

            if train_cfg.type == 'IRMTrainer':
                model = [create_model(model_cfg), create_model(model_cfg)]
            elif train_cfg.type == 'LLALTrainer':
                model = [create_model(model_cfg), create_module(cfg.additional_module)]
            elif train_cfg.type == 'TiDALTrainer':
                model = [create_model(model_cfg), create_module(cfg.pred_module)]
            elif train_cfg.type == 'TQSTrainer':
                model = [create_model(model_cfg), create_module(cfg.multi_classify), create_module(cfg.discriminator)]
            elif train_cfg.type == 'MultiClassifierTrainer' or train_cfg.type == 'RecallMultiClassifierTrainer':
                cfg.multi_classifier.n_classifier = train_ds_num
                model = [create_model(model_cfg), create_module(cfg.multi_classifier)]
            elif train_cfg.type == 'DiscriminatorMultiClassifierTrainer':
                cfg.multi_classifier.n_classifier = train_ds_num
                cfg.domain_discriminator.num_domains = train_ds_num
                model = [create_model(model_cfg), create_module(cfg.domain_discriminator), create_module(cfg.multi_classifier), create_module(cfg.multi_discriminator)]
            elif train_cfg.type == 'DANNTrainer':
                cfg.domain_discriminator.num_domains = train_ds_num
                model = [create_model(model_cfg), create_module(cfg.domain_discriminator)]
            elif train_cfg.type == 'DANNMultiClassifierTrainer':
                cfg.multi_classifier.n_classifier = train_ds_num
                cfg.domain_discriminator.num_domains = train_ds_num
                model = [create_model(model_cfg), create_module(cfg.domain_discriminator),
                         create_module(cfg.multi_classifier)]
            elif train_cfg.type == 'TMSALTrainer' or train_cfg.type == 'MarginTMSALTrainer' or train_cfg.type == 'MarginTMSALTrainer_A':
                cfg.multi_classifier.n_classifier = train_ds_num
                model = [create_model(model_cfg), create_module(cfg.multi_classifier)]
            else:
                model = [create_model(model_cfg)]
            train_type = globals()[train_cfg.type]
            train_cfg.output_dir = cfg.output_dir
            trainer = train_type(model, train_cfg, label_info, train_ds, select_ds, test_ds, writer, strategy_cfg, cfg.amp, logger)

            # train
            trainer_checkpoint = os.path.join(cfg.output_dir, 'checkpoint_trainer_rd_%d.pt'%rd)
            start_time = time.time()

            if os.path.exists(trainer_checkpoint):
                print('load check point')
                trainer.load_state_dict(torch.load(trainer_checkpoint))
                train_time = 0.
            else:
                best_acc = trainer.train(str(rd))
                train_time = time.time() - start_time

                if train_cfg.get("is_multi_classifier", False) == True:
                    result, _ = trainer.multi_classifier_test()
                else:
                    result = trainer.base_model_accuracy()
                logger.info(result)
                cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                print(cur_time)
                print('final: ',result)
                cur_acc = best_acc
                strategy_cfg.cur_acc = cur_acc
                # total_acc = result['current']
                acc[rd] = cur_acc
                print('Round {}\ntesting accuracy {}'.format(rd, acc[rd]))
                # torch.save(model.state_dict(), checkpoint_path)
                # torch.save(trainer.center_loss.state_dict(), trainer_checkpoint)
                torch.save(trainer.state_dict(), trainer_checkpoint)

            if rd == active_cfg.n_round:
                result_writer.writerow([acc[rd], train_time, 0.])
                result_file.flush()
                continue


            # query
            print('start query')
            print('selecting samples from unlabeled:', len(label_info.unlabel_ind))
            strategy_name = strategy_cfg.type
            strategy = query_strategy[strategy_name](net=model,
                                                     active_cfg=active_cfg,
                                                     strategy_cfg=strategy_cfg,
                                                     label_info=label_info,
                                                     select_ds=select_ds,
                                                     trainer=trainer)
            select_ind = strategy.query(active_cfg.n_select)
            duration = time.time() - start_time
            if cfg.debug:
                continue
            # update
            # label_ind = label_info.label_ind
            # unlabel_ind = label_info.unlabel_ind
            select_ind = np.array(select_ind)
            label_ind = np.append(label_ind, select_ind)
            label_ind = np.unique(label_ind)
            label_ind = np.random.permutation(label_ind)
            unlabel_ind = np.setdiff1d(total_train_ind, label_ind)
            label_info.label_ind = label_ind
            label_info.unlabel_ind = unlabel_ind
            print(len(label_ind), len(select_ind))
            assert len(label_ind) == active_cfg.n_initial + rd * active_cfg.n_select

            # cumsum = select_ds.cumsum()
            # print(cumsum)
            # labeled_count = torch.zeros(train_ds_num+1)
            # for i in range(train_ds_num+1):
            #     labeled_count[i] = torch.sum((torch.from_numpy(label_ind) >= cumsum[i]) & (torch.from_numpy(label_ind) < cumsum[i+1]))
            # print(labeled_count)
            print('save ind')
            # exit()
            # label_dist = torch.zeros(train_cfg.dataset_number)
            # domain_lens = len(train_ds.domain_lens)
            # # train_ds.ds_ind
            # dist_count = 0
            # for i in range(train_cfg.dataset_number):
            #     if i == 0:
            #         label_dist[i] = label_ind < domain_lens[0]
            #     elif i == domain_lens-1:
            #         label_dist[i] = label_ind >= dist_count
            #     el

            torch.save(torch.tensor(label_ind), labeled_ind_path)

            # record
            writer.add_scalar('test_accuracy', acc[rd], rd)
            result_writer.writerow([acc[rd], train_time, duration])
            result_file.flush()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification active learning")
    parser.add_argument('--config', type=str)
    # parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--amp', default=False, type=bool)
    parser.add_argument('--debug', default=False)

    args, _ = parser.parse_known_args()
    cfg = Config.fromfile(args.config)

    # cfg.data_dir=args.data_dir
    cfg.output_dir=args.output_dir
    cfg.seed=args.seed
    cfg.amp=args.amp
    cfg.debug=args.debug

    set_seeds(cfg.seed)
    active_train(cfg)

    # if args.strategy == 'All':
    #     for strategy in query_strategy.keys():
    #         args.strategy = strategy
    #         active_train(args)
    # else:
    #     active_train(arg