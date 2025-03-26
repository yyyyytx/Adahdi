export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active



config_prefix="configs/caltech101/swinv2_"
output_prefix="outputs/caltech101_swinv2_"
file_format=".py"
bottom="_"
#for ADA_AL in 'tmsal' 'coreset' 'alpha_mix' 'grad_norm' 'sdm' 'active_ft' 'las' 'mhpl' 'random' 'llal'; do
#for ADA_AL in 'alpha_mix' 'entropy' 'grad_norm' 'metric' 'random' 'real' 'noise'; do
for ADA_AL in 'tidal'; do
  for s in 1 2; do

    config=${config_prefix}${ADA_AL}${file_format}
    out_dir=${output_prefix}${ADA_AL}@${s}

    echo $config
    echo $out_dir

    echo $DIR
    if [ -d "$out_dir" ]; then
      echo " The results exist at ${DIR}"
    else
      python main.py --config $config --output_dir $out_dir --seed $s
    fi
  done
done


config_prefix="configs/caltech256/swinv2_"
output_prefix="outputs/caltech256_swinv2_"
file_format=".py"
bottom="_"
#for ADA_AL in 'tmsal' 'coreset' 'alpha_mix' 'grad_norm' 'sdm' 'active_ft' 'las' 'mhpl' 'random' 'llal'; do
#for ADA_AL in 'active_ft' 'alpha_mix' 'entropy' 'grad_norm' 'metric' 'random' 'real'; do
for ADA_AL in 'tidal'; do

  for s in 1 2; do
    config=${config_prefix}${ADA_AL}${file_format}
    out_dir=${output_prefix}${ADA_AL}@${s}

    echo $config
    echo $out_dir

    echo $DIR
    if [ -d "$out_dir" ]; then
      echo " The results exist at ${DIR}"
    else
      python main.py --config $config --output_dir $out_dir --seed $s
    fi
  done
done




