DATASET=/path/to/dataset
export CUDA_VISIBLE_DEVICES=0

NAME="Name_of_the_folder_to_save"
PYTHONPATH=$CODE python train.py --name $NAME --arch res50nodown --cropSize 96  --norm_type resnet --resize_size 256 --resize_ratio 0.75 --blur_sig 0.0,3.0 --cmp_method cv2,pil --cmp_qual 30,100 --resize_prob 0.2 --jitter_prob  0.8 --colordist_prob 0.2 --cutout_prob 0.2 --noise_prob 0.2 --blur_prob 0.5 --cmp_prob 0.5 --rot90_prob 1.0 --dataroot $DATASET