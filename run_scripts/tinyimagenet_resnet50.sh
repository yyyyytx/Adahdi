export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active



config_prefix="configs/tinyimagenet/resnet50_"
output_prefix="outputs/tinyimagenet_resnet50_"
file_format=".py"
#for ADA_AL in 'active_ft' 'alpha_mix' 'entropy' 'gcn' 'grad_norm' 'llal' 'metric' 'random' 'real'; do
for ADA_AL in 'noise'; do
  for s in 3; do
    config=${config_prefix}${ADA_AL}${file_format}
    out_dir=${output_prefix}${ADA_AL}@${s}
    echo $config
    echo $out_dir

    python main.py --config $config --output_dir $out_dir --seed $s
  done
done

