export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active



config_prefix="configs/domainnet/resnet50"
output_prefix="outputs/domainnet_resnet50"
format=".py"
bottom="_"
test="test"

# for ADA_AL in 'sdm' 'active_ft' 'tmsal' 'random' 'coreset' 'grad_norm' 'llal'; do

#for ADA_AL in 'active_ft' 'tmsal' 'random' 'coreset' 'grad_norm' 'llal'; do
for ADA_AL in 'mhpl'; do
  for s in 1 ; do
      for domain in 'domain234561'; do
      config=${config_prefix}${bottom}${ADA_AL}${bottom}${domain}${format}
      out_dir=${output_prefix}${bottom}${domain}${bottom}${ADA_AL}${bottom}${test}@${s}
      echo $config
      echo $out_dir

      python main.py --config $config --output_dir $out_dir --seed $s
      done
  done
done



