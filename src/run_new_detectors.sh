#!/bin/bash
# Evaluate RAID (PGD) and RAID-APGD - full ELSA_TEST, 7-model ensemble,
# eps 8/16/32 - with the evaluation-only detectors (DINOv2 / ViT-T
# probes, Effort, AIDE, D3, OmniAID, DDA), plus their full clean ELSA baseline (Table 5 of
# build_results_table.py). Existing results are skipped. The original 7
# detectors are not handled here (compute_metrics.py --mode clean/experiments).
#
# Missing datasets are reported, not generated. Generate with e.g.:
#   ./run_experiments.sh <device> 16/255 0.03 --mode dataset-generation --attack apgd
#
# Usage:
#   source env_set.sh
#   ./run_new_detectors.sh <device> [eps ...] [--clean] [--pgd_only] [--detectors a,b]
#
#   eps          budgets to evaluate (default: 8 16 32), to split across GPUs
#   --clean      also run the clean full-ELSA baseline - pass it to exactly
#                one of the parallel runs
#   --pgd_only   RAID (PGD) only, skip RAID-APGD. Default: both
#   --detectors  comma-separated subset of the evaluation-only detectors
#                below. Default: all of them
#
#   ./run_new_detectors.sh cuda:0 8 --clean &
#   ./run_new_detectors.sh cuda:1 16 32 &
#   ./run_new_detectors.sh cuda:0 8 --clean --detectors omniaid,dda &

set -euo pipefail
cd "$(dirname "$0")"

DEVICE="${1:?Usage: $0 <device> [eps ...] [--clean] [--pgd_only] [--detectors a,b]}"
shift
EPS=() CLEAN=0 ATTACKS=(raw apgd) ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --clean)     CLEAN=1 ;;
        --pgd_only)  ATTACKS=(raw) ;;
        --detectors) ONLY="${2:?--detectors needs a comma-separated list}"; shift ;;
        *)           EPS+=("$1") ;;
    esac
    shift
done
[ ${#EPS[@]} -gt 0 ] || { EPS=(8 16 32); CLEAN=1; }
EPS_ARGS=(); for e in "${EPS[@]}"; do EPS_ARGS+=("$e/255"); done

NEW=(vit_lp14_dinov2 vit_lp14_reg_dinov2
     vit_tp16_224_augreg_in21k vit_tp16_224_code_augreg_in21k
     effort aide d3 omniaid dda)
if [ -n "$ONLY" ]; then
    IFS=, read -ra SEL <<< "$ONLY"
    for d in "${SEL[@]}"; do
        [[ " ${NEW[*]} " == *" $d "* ]] || { echo "Unknown detector '$d' (choose from: ${NEW[*]})"; exit 1; }
    done
    NEW=("${SEL[@]}")
fi
echo "Detectors: ${NEW[*]}"

# 0. Check the 6 expected datasets
STEP=([8]="10_0.01" [16]="10_0.03" [32]="10_0.03")
ENS="['ojha2023', 'corvi2023', 'cavia2024', 'chen2024_convnext', 'chen2024_clip', 'koutlis2024', 'wang2020']"
for prefix in "${ATTACKS[@]}"; do
    for eps in "${EPS[@]}"; do
        pkl="${OUTPUT_DIR}/ADV/ELSA_TEST/adv_${prefix}_${ENS}_${eps}_255_${STEP[$eps]}/adv_dataset.pkl"
        [ -f "$pkl" ] || echo "WARNING: missing ${prefix} eps=${eps}/255 dataset - its cells will be skipped: $pkl"
    done
done

# 1. Clean baseline on the full ELSA_TEST -> clean/ELSA_full (ASR denominator)
if [ "$CLEAN" = 1 ]; then
    python3 compute_metrics.py --mode clean --subset -1 --datasets ELSA_TEST \
        --detectors "${NEW[@]}" --device "$DEVICE"
fi

# 2. New detectors on RAID + RAID-APGD -> output/ADV/ELSA_TEST/<adv>/<model>/
python3 compute_metrics.py --mode experiments --datasets ELSA_TEST \
    --detectors "${NEW[@]}" --epsilons "${EPS_ARGS[@]}" --attacks "${ATTACKS[@]}" --device "$DEVICE"

# 3. Tables (Table 5 = RAID dataset, all detectors)
python3 build_results_table.py --dataset elsa
