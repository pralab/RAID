#!/bin/bash
# Held-out detector on every leave-one-out PGD folder (ELSA_TEST_1000) with
# resize64 / crop64 / flip / jpeg75 applied to the adversarial images
# (compute_metrics.py --mode transforms). One budget per GPU.
#
# Usage (inside tmux):
#   bash src/run_transforms.sh <epsilon> <device>
#   bash src/run_transforms.sh 32/255 cuda:0
#   bash src/run_transforms.sh 16/255 cuda:1
# Log: src/transforms_<eps>.log

set -eo pipefail

EPS="${1:?epsilon, e.g. 32/255}"
DEVICE="${2:?device, e.g. cuda:0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate raid
cd "$(dirname "$0")"
source env_set.sh

LOG="transforms_${EPS//\//_}.log"
echo "[$(date '+%F %T')] transforms eps=$EPS device=$DEVICE -> $LOG" | tee "$LOG"

python3 compute_metrics.py --mode transforms \
    --epsilons "$EPS" \
    --device "$DEVICE" \
    --batch_size 32 \
    --num_workers 8 \
    2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] done eps=$EPS" | tee -a "$LOG"
