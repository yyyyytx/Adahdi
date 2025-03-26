export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active



config_prefix="configs/officehome/resnet50"
output_prefix="outputs/officehome_resnet50"
format=".py"
bottom="_"
new="new"
test="test"
#for ADA_AL in 'coreset' 'alpha_mix' 'grad_norm' 'sdm' 'active_ft' 'las' 'tmsal' 'mhpl' 'random' 'llal'; do
for ADA_AL in 'tmsal'; do
  for s in 1 2 3; do
      for domain in 'domain1234' 'domain2341' 'domain3412' 'domain4123'; do
      config=${config_prefix}${bottom}${ADA_AL}${bottom}${domain}${format}
      out_dir=${output_prefix}${bottom}${domain}${bottom}${ADA_AL}${new}@${s}
      echo $config
      echo $out_dir

      python main.py --config $config --output_dir $out_dir --seed $s
      done
  done
done



