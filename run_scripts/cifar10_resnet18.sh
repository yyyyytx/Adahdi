export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active

#config=configs/cifar10/resnet18_tidal.py
#base_dir=outputs/cifar10_resnet18_tidal
#
#seed=1
#python main.py --config $config --output_dir "${base_dir}@${seed}" --seed $seed
#seed=2
#python main.py --config $config --output_dir "${base_dir}@${seed}" --seed $seed
#seed=3
#python main.py --config $config --output_dir "${base_dir}@${seed}" --seed $seed

#config=configs/cifar10/resnet_test.py
#base_dir=outputs/cifar10_resnet18_test
#
#seed=1
#python main.py --config $config --output_dir "${base_dir}@${seed}" --seed $seed


config_prefix="configs/cifar10/resnet18_"
output_prefix="outputs/cifar10_resnet18_"
file_format=".py"
#for ADA_AL in 'active_ft' 'alpha_mix' 'entropy' 'gcn' 'grad_norm' 'llal' 'metric' 'random' 'real; do
for ADA_AL in 'noise'; do
  for s in 1 2 3; do
    config=${config_prefix}${ADA_AL}${file_format}
    out_dir=${output_prefix}${ADA_AL}@${s}
    echo $config
    echo $out_dir
    python main.py --config $config --output_dir $out_dir --seed $s
  done
done

