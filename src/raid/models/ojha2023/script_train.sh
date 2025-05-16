DATASET=data/ELSA_D3

python src/raid/models/ojha2023/train.py --name=clip_vitl14 --elsa_d3_data_path=$DATASET --data_mode=elsa_d3  --arch=CLIP:ViT-L/14  --fix_backbone