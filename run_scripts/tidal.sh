export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active



#config_prefix="configs/domainnet/resnet50"
#output_prefix="outputs/domainnet_resnet50"
#format=".py"
#bottom="_"
#for ADA_AL in 'tidal'; do
#  for s in 1 ; do
#      for domain in 'domain234561' 'domain345612' 'domain456123' 'domain432165'; do
#      config=${config_prefix}${bottom}${ADA_AL}${bottom}${domain}${format}
#      out_dir=${output_prefix}${bottom}${domain}${bottom}${ADA_AL}@${s}
#      echo $config
#      echo $out_dir
#
#      python main.py --config $config --output_dir $out_dir --seed $s
#      done
#  done
#done

#
#config_prefix="configs/officehome/swinv2"
#output_prefix="outputs/officehome_swinv2"
#format=".py"
#bottom="_"
#for ADA_AL in 'tidal'; do
#  for s in 1 2 3; do
#      for domain in 'domain1234' 'domain2341' 'domain3412' 'domain4123'; do
#      config=${config_prefix}${bottom}${ADA_AL}${bottom}${domain}${format}
#      out_dir=${output_prefix}${bottom}${domain}${bottom}${ADA_AL}@${s}
#      echo $config
#      echo $out_dir
#
#      python main.py --config $config --output_dir $out_dir --seed $s
#      done
#  done
#done



config_prefix="configs/pacs/resnet50"
output_prefix="outputs/pacs_resnet50"
format=".py"
bottom="_"
for ADA_AL in 'tidal'; do
  for s in 1 2 3; do
      for domain in 'domain1234' 'domain2341' 'domain3412' 'domain4123'; do
      config=${config_prefix}${bottom}${ADA_AL}${bottom}${domain}${format}
      out_dir=${output_prefix}${bottom}${domain}${bottom}${ADA_AL}@${s}
      echo $config
      echo $out_dir

      python main.py --config $config --output_dir $out_dir --seed $s
      done
  done
done