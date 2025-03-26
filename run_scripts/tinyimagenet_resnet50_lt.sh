export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active



config_prefix="configs/tinyimagenet/resnet50_"
output_prefix="outputs/tinyimagenet_resnet50_"
file_format=".py"
#for ADA_AL in 'active_ft_lt' 'alpha_mix_lt' 'entropy_lt' 'grad_norm_lt' 'llal_lt' 'metric_lt' 'random_lt' 'real_lt'; do
for ADA_AL in 'tidal_lt'; do
  for s in 1 2; do
    config=${config_prefix}${ADA_AL}${file_format}
    out_dir=${output_prefix}${ADA_AL}@${s}
    echo $config
    echo $out_dir

    python main.py --config $config --output_dir $out_dir --seed $s
  done
done

