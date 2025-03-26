export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active



config_prefix="configs/officehome/resnet50"
output_prefix="outputs/officehome_resnet50"
format=".py"
bottom="_"
for ADA_AL in 'aba_alpha_mix' 'aba_active_ft' 'aba_las' 'aba_mhpl' 'aba_margin' 'aba_duc' 'aba_sdm' 'aba_coreset' 'aba_noise' 'aba_mada'; do
  for s in 1 ; do
      for domain in 'domain1234'; do
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