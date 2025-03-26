export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active

config=configs/caltech256/resnet50_tidal.py
base_dir=outputs/caltech256_resnet50_tidal

seed=1
python main.py --config $config --output_dir "${base_dir}@${seed}" --seed $seed
seed=2
python main.py --config $config --output_dir "${base_dir}@${seed}" --seed $seed
seed=3
python main.py --config $config --output_dir "${base_dir}@${seed}" --seed $seed
