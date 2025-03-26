export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active

config_prefix="configs/cifar100/resnet50_"
output_prefix="outputs/cifar100_resnet50_"
file_format=".py"
for ADA_AL in 'weak_coreset' 'strong_coreset' 'weak_active_ft' 'strong_active_ft' 'weak_entropy' 'strong_entropy' 'weak_margin' 'strong_margin' 'weak_random' 'strong_random' 'weak_alpha_mix' 'strong_alpha_mix'; do
  for s in 1; do
    config=${config_prefix}${ADA_AL}${file_format}
    out_dir=${output_prefix}${ADA_AL}@${s}
    echo $config
    echo $out_dir

    python main.py --config $config --output_dir $out_dir --seed $s
  done
done


#config_prefix="configs/cifar100/resnet18_"
#output_prefix="outputs/cifar100_resnet18_"
#file_format=".py"
#for ADA_AL in 'active_ft' 'alpha_mix' 'entropy' 'gcn' 'grad_norm' 'llal' 'metric' 'random' 'real; do
#  for s in 1 2 3; do
#    config=${config_prefix}${ADA_AL}${file_format}
#    out_dir=${output_prefix}${ADA_AL}@${s}
#    echo $config
#    echo $out_dir
#
#    python main.py --config $config --output_dir $out_dir --seed $s
#  done
#done

