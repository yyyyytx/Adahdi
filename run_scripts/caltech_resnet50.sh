export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active



config_prefix="configs/caltech101/resnet50_"
output_prefix="outputs/caltech101_resnet50_"
file_format=".py"
bottom="_"
#for ADA_AL in 'tmsal' 'coreset' 'alpha_mix' 'grad_norm' 'sdm' 'active_ft' 'las' 'mhpl' 'random' 'llal'; do
for ADA_AL in 'noise'; do

  for s in 1 2 3; do
    config=${config_prefix}${ADA_AL}${file_format}
    out_dir=${output_prefix}${ADA_AL}@${s}
    echo $config
    echo $out_dir

    python main.py --config $config --output_dir $out_dir --seed $s
  done
done


config_prefix="configs/caltech256/resnet50_"
output_prefix="outputs/caltech256_resnet50_"
file_format=".py"
bottom="_"
#for ADA_AL in 'tmsal' 'coreset' 'alpha_mix' 'grad_norm' 'sdm' 'active_ft' 'las' 'mhpl' 'random' 'llal'; do
for ADA_AL in 'noise'; do

  for s in 1 2 3; do
    config=${config_prefix}${ADA_AL}${file_format}
    out_dir=${output_prefix}${ADA_AL}@${s}
    echo $config
    echo $out_dir

    python main.py --config $config --output_dir $out_dir --seed $s
  done
done




