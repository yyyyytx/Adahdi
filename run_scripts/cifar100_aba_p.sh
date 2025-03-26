export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active


config_prefix="configs/cifar100/resnet18_metric_"
output_prefix="outputs/cifar100_resnet18_metric"
file_format=".py"
#for ADA_AL in 'aba_p1' 'aba_p2' 'aba_p3' 'aba_p4' 'aba_p5' 'aba_p6' 'aba_p7' 'aba_p8' 'aba_p9' 'aba_p10'; do
for ADA_AL in 'aba_p9' 'aba_p10'; do
  for s in 1; do
    config=${config_prefix}${ADA_AL}${file_format}
    out_dir=${output_prefix}${ADA_AL}@${s}
    echo $config
    echo $out_dir

    python main.py --config $config --output_dir $out_dir --seed $s
  done
done

