export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active



config_prefix="configs/domainnet/resnet50"
output_prefix="outputs/domainnet_resnet50"
format=".py"
bottom="_"

# for ADA_AL in 'sdm' 'active_ft' 'tmsal' 'random' 'coreset' 'grad_norm' 'llal'; do

#for ADA_AL in 'active_ft' 'tmsal' 'random' 'coreset' 'grad_norm' 'llal'; do
#for ADA_AL in 'las'; do
#for ADA_AL in 'active_ft' 'coreset' 'las' 'mhpl' 'tmsal' 'random' 'alpha_mix'; do
#  for s in 1 ; do
#      for domain in 'domain234561'; do
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
#for ADA_AL in 'active_ft' 'grad_norm' 'las' 'mhpl' 'random' 'alpha_mix'; do
#  for s in 1 ; do
#      for domain in 'domain345612'; do
#      config=${config_prefix}${bottom}${ADA_AL}${bottom}${domain}${format}
#      out_dir=${output_prefix}${bottom}${domain}${bottom}${ADA_AL}@${s}
#      echo $config
#      echo $out_dir
#
#      python main.py --config $config --output_dir $out_dir --seed $s
#      done
#  done
#done

#for ADA_AL in 'llal' 'las' 'mhpl' 'random'; do
#  for s in 1 ; do
#      for domain in 'domain432165'; do
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
#for ADA_AL in 'active_ft' 'las' 'mhpl' 'random' 'alpha_mix'; do
#  for s in 1 ; do
#      for domain in 'domain456123'; do
#      config=${config_prefix}${bottom}${ADA_AL}${bottom}${domain}${format}
#      out_dir=${output_prefix}${bottom}${domain}${bottom}${ADA_AL}@${s}
#      echo $config
#      echo $out_dir
#
#      python main.py --config $config --output_dir $out_dir --seed $s
#      done
#  done
#done


for ADA_AL in 'mada'; do
  for s in 1 ; do
      for domain in 'domain234561' 'domain345612' 'domain456123' 'domain432165'; do
      DIR=${output_prefix}${bottom}${domain}${bottom}${ADA_AL}@${s}
        echo $DIR
        if [ -d "$DIR" ]; then
          echo " The results exist at ${DIR}"
        else
          config=${config_prefix}${bottom}${ADA_AL}${bottom}${domain}${format}
          echo $config
          echo $out_dir
          python main.py --config $config --output_dir $DIR --seed $s
        fi
      done
  done
done







