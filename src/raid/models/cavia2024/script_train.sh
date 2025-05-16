# Corvi training

DATASET=data/ELSA_D3_1000
export CUDA_VISIBLE_DEVICES=0

NAME="train_test"
python src/raid/models/cavia2024/train.py --mode binary_elsa --name LaDeDa --dataroot $DATASET --checkpoints_dir ./data/checkpoints --batch_size 32 --lr 0.0002 --delr_freq 10