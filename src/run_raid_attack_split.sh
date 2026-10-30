#!/bin/bash
# Full RAID split (entire ELSA_TEST, full 7-detector ensemble) for one attack
# and one budget, then PNG export (<adv dir>/IMAGE_DATASET/). Same K / alpha as
# the PGD release: K=10, alpha=0.01 at 8/255 and 0.03 at 16/255 and 32/255.
# Skips generation if adv_dataset.pkl already exists (APGD is already generated,
# so for APGD this only exports the PNGs).
#
# Usage (inside tmux, one budget per GPU):
#   bash src/run_raid_attack_split.sh <attack> <epsilon> <device> [batch_size]
#   bash src/run_raid_attack_split.sh cwa 8/255  cuda:0
#   bash src/run_raid_attack_split.sh cwa 16/255 cuda:1
#   bash src/run_raid_attack_split.sh cwa 32/255 cuda:2
#   bash src/run_raid_attack_split.sh apgd 16/255 cuda:0   # PNG export only
# Log: src/raid_<attack>_<eps>.log

set -eo pipefail

ATTACK="${1:?attack: cwa | apgd}"
EPS="${2:?epsilon, e.g. 16/255}"
DEVICE="${3:?device, e.g. cuda:0}"
BATCH_SIZE="${4:-4}"

case "$EPS" in
    8/255) STEP_SIZE=0.01 ;;
    16/255|32/255) STEP_SIZE=0.03 ;;
    *) echo "unsupported epsilon $EPS"; exit 1 ;;
esac
NUM_STEPS=10
# Same order as the PGD release (defines the folder name).
MODELS=(ojha2023 corvi2023 cavia2024 chen2024_convnext chen2024_clip koutlis2024 wang2020)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate raid
cd "$(dirname "$0")"
source env_set.sh
# attack_generate.py imports both `attacks` (raid/) and `raid.attacks` (src/)
export PYTHONPATH="$PWD:$PWD/raid:${PYTHONPATH}"

LOG="raid_${ATTACK}_${EPS//\//_}.log"
MODELS_STR="['$(IFS=,; echo "${MODELS[*]}" | sed "s/,/', '/g")']"
ADV_DIR="${OUTPUT_DIR}/ADV/$(basename "$TEST_PATH")/adv_${ATTACK}_${MODELS_STR}_${EPS//\//_}_${NUM_STEPS}_${STEP_SIZE}"
DEVICES=(); for _ in "${MODELS[@]}"; do DEVICES+=("$DEVICE"); done

{
echo "[$(date '+%F %T')] RAID ${ATTACK} split eps=${EPS} alpha=${STEP_SIZE} K=${NUM_STEPS} device=${DEVICE} batch=${BATCH_SIZE}"
echo "adv dir: ${ADV_DIR}"

if [ -f "${ADV_DIR}/adv_dataset.pkl" ]; then
    echo "[$(date '+%F %T')] adv_dataset.pkl exists, skipping generation"
else
    python raid/attack_generate.py \
        --attack "$ATTACK" \
        --models "${MODELS[@]}" \
        --device "${DEVICES[@]}" \
        --path_to_dataset "$TEST_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --epsilon "$EPS" \
        --num_steps "$NUM_STEPS" \
        --step_size "$STEP_SIZE" \
        --subset -1 \
        --generate
fi

echo "[$(date '+%F %T')] PNG export -> ${ADV_DIR}/IMAGE_DATASET"
python raid/scripts/generate_image_dataset.py \
    --path_to_dataset "$TEST_PATH" \
    --path_to_adv_dataset "${ADV_DIR}/adv_dataset.pkl" \
    --device "$DEVICE" \
    | grep -v "_adv.png$"

for c in "${ADV_DIR}"/IMAGE_DATASET/*/; do
    echo "  $(basename "$c"): $(ls "$c" | wc -l) PNGs"
done
echo "[$(date '+%F %T')] done"
} 2>&1 | tee -a "$LOG"
