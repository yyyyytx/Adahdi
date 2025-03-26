export PYTHONPATH=$PYTHONPATH:/media/liu/data/debiased_irm_active

config=configs/caltech101/resnet50_metric.py
base_dir=outputs/caltech101_resnet50_metric

seed=1
python main.py --config $config --output_dir "${base_dir}@${seed}" --seed $seed
seed=2
python main.py --config $config --output_dir "${base_dir}@${seed}" --seed $seed
seed=3
python main.py --config $config --output_dir "${base_dir}@${seed}" --seed $seed
