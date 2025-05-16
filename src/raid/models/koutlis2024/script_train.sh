# koutlis2024 training
# runs a hyperparamter search; in the end displays results for all trials, you should pick the best checkpoint
# TODO: fix ELSA paths and add logic for train/val split in get_loaders() and TrainingDatasetELSAD3()
# I changed get_transforms() to use pad_if_needed, since not all images have size 224x224, which caused an exception

export CUDA_VISIBLE_DEVICES=0

python src/raid/models/koutlis2024/train.py
