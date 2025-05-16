# Run the PGD attack on the ensemble model and generate the adversarial dataset

# Available Models
# ojha2023, corvi2023, cavia2024, chen2024_convnext, chen2024_clip, koutlis2024, wang2020
# Available Ensemble Models
# raw, avg, random

for model in cavia2024 chen2024_convnext chen2024_clip corvi2023 koutlis2024 ojha2023 wang2020; do
    python src/raid/attack_generate.py \
    --ensembling_strategy raw \
    --models $model \
    --path_to_dataset data/RAID/<select_subfolder> \
    --output_dir <output_dir> \
    --batch_size 8 \
    --epsilon 32/255 \
    --num_steps 10 \
    --generate \
    --step_size 5e-2 \
    --subset -1
done



#    --generate \
#    --eval_model corvi2023 \
#    --eval_output_dir output/evaluate_detector \
