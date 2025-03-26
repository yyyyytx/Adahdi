export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active



config_prefix="configs/officehome/resnet50"
output_prefix="outputs/officehome_resnet50"
format=".py"
bottom="_"
#for ADA_AL in 'tmsal' 'coreset' 'alpha_mix' 'grad_norm' 'sdm' 'active_ft' 'las' 'mhpl' 'random' 'llal'; do
#for ADA_AL in 'noise'; do
for ADA_AL in 'mada'; do
  for s in 1 2 3; do
#      for domain in 'domain1234'; do

      for domain in 'domain1234' 'domain2341' 'domain3412' 'domain4123'; do
        DIR=${output_prefix}${bottom}${domain}${bottom}${ADA_AL}@${s}
        echo $DIR
        if [ -d "$DIR" ]; then
          echo " The results exist at ${DIR}"
        else
          config=${config_prefix}${bottom}${ADA_AL}${bottom}${domain}${format}
          echo $config
          python main.py --config $config --output_dir $DIR --seed $s
        fi

      done
  done
done


#for ADA_AL in 'vessal'; do
#  for s in 1 ; do
#      for domain in 'domain1234'; do
#        DIR=${output_prefix}${bottom}${domain}${bottom}${ADA_AL}@${s}
#        echo $DIR
#        if [ -d "$DIR" ]; then
#          echo " The results exist at ${DIR}"
#        else
#          config=${config_prefix}${bottom}${ADA_AL}${bottom}${domain}${format}
#          echo $config
#          python main.py --config $config --output_dir $DIR --seed $s
#        fi
#
#      done
#  done
#done



