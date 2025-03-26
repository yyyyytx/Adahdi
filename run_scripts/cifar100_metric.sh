export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active


config_prefix="configs/cifar100/resnet18_metric_"
output_prefix="outputs/cifar100_resnet18_metric"
file_format=".py"
for ADA_AL in 'aba_gamma' 'aba_gamma1' 'aba_gamma2' 'aba_gamma3' 'aba_gamma4' 'aba_gamma5'; do
  for s in 1 2 3; do
    config=${config_prefix}${ADA_AL}${file_format}
    out_dir=${output_prefix}${ADA_AL}@${s}
    echo $config
    echo $out_dir

    python main.py --config $config --output_dir $out_dir --seed $s
  done
done

