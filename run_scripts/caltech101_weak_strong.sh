export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active

config_prefix="configs/caltech101/resnet50_"
output_prefix="outputs/caltech101_resnet50_"
file_format=".py"
for ADA_AL in 'weak_random' 'weak_margin' 'weak_entropy' 'weak_alpha_mix' 'weak_active_ft' 'strong_random' 'strong_margin' 'strong_entropy' 'strong_alpha_mix' 'strong_active_ft'; do
  for s in 1 2 3; do
    config=${config_prefix}${ADA_AL}${file_format}
    out_dir=${output_prefix}${ADA_AL}@${s}
    echo $config
    echo $out_dir

    python main.py --config $config --output_dir $out_dir --seed $s
  done
done