export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active



config_prefix="configs/officehome/swinv2"
output_prefix="outputs/officehome_swinv2"
format=".py"
bottom="_"
new="new"
#for ADA_AL in 'coreset' 'grad_norm' 'alpha_mix' 'sdm' 'active_ft' 'las' 'tmsal' 'mhpl' 'random'; do
for ADA_AL in 'tmsal'; do
  for s in 1; do
      for domain in 'domain1234'; do
      config=${config_prefix}${bottom}${ADA_AL}${bottom}${domain}${format}
      out_dir=${output_prefix}${bottom}${domain}${bottom}${ADA_AL}${new}@${s}
      echo $config
      echo $out_dir

      python main.py --config $config --output_dir $out_dir --seed $s
      done
  done
done



