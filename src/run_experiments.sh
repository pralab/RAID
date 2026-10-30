#!/bin/bash
# Single entry point for all GPU attack-generation experiments: the PGD
# ablation sweep, and the white-box / leave-one-out / full-RAID ensemble
# attacks, for one device.
#
# Usage:
#   source env_set.sh
#   ./run_experiments.sh <device> [epsilon] [step_size] [num_steps] [--mode ...] [--epsilon E] [--step_size S] [--num_steps N] [--seeds "s1 s2 ..."] [--sub_subset M]
#
# Modes:
#   experiments (default) - white-box (1000), leave-one-out (1000), full RAID
#                            ensemble (full dataset) for one (epsilon, step_size)
#   ablation               - PGD hyperparameter sweep (steps x alpha x seeds)
#                            for one epsilon, leave-one-out ensembles, subset 200
#   subsampling            - Seeded experiments (--seeds, default "42 123 234 345 456",
#                            --sub_subset samples each) for variance estimation.
#   additional             - APGD and CWA leave-one-out attacks (1000 images)
#                            for one (epsilon, step_size), matching PGD's own
#                            epsilon/steps/step_size for a like-for-like
#                            comparison (Table B.6-style replication).
#   additional_generators  - white-box and leave-one-out (--gen_attacks,
#                            default "wb loo") on the 11
#                            external generators in $ADDGEN_SRC not in ELSA_TEST,
#                            200 images each (--sub_subset), real = ELSA_TEST/real,
#                            for one (epsilon, step_size)
#   dataset-generation     - Full 7-model ensemble attack on the full dataset
#                            for a given --attack (pgd|apgd|cwa). Skips
#                            white-box and LOO (already done in additional).
#   all                    - ablation then experiments, for one epsilon
#
# Examples (3 GPUs, 3 budgets):
#   ./run_experiments.sh cuda:0 8/255  0.01 --mode all
#   ./run_experiments.sh cuda:1 16/255 0.03 --mode all
#   ./run_experiments.sh cuda:2 32/255 0.03 --mode all
#
#   ./run_experiments.sh cuda:0 --mode ablation --epsilon 8/255
#   ./run_experiments.sh cuda:0 8/255 0.01 --mode experiments
#   ./run_experiments.sh cuda:0 8/255 0.01 --mode dataset-generation --attack apgd
#   ./run_experiments.sh cuda:0 8/255 0.01 --mode additional_generators
#
# Machine-specific paths are NOT set here - see env_set.sh.example.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/raid:${PYTHONPATH:-}"

DEVICE="${1:?Usage: $0 <device> [epsilon] [step_size] [num_steps] [--mode all|ablation|experiments|subsampling|additional]}"
shift

MODE="experiments"
EPSILON=""
STEP_SIZE=""
NUM_STEPS=10
ATTACK="pgd"
SUB_SEEDS="42 123 234 345 456"
SUB_SUBSET=200
GEN_ATTACKS="wb loo"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2"; shift 2 ;;
        --epsilon)    EPSILON="$2"; shift 2 ;;
        --step_size)  STEP_SIZE="$2"; shift 2 ;;
        --num_steps)  NUM_STEPS="$2"; shift 2 ;;
        --attack)     ATTACK="$2"; shift 2 ;;
        --seeds)      SUB_SEEDS="$2"; shift 2 ;;
        --sub_subset) SUB_SUBSET="$2"; shift 2 ;;
        --gen_attacks) GEN_ATTACKS="$2"; shift 2 ;;
        *)
            if [[ -z "$EPSILON" ]]; then
                EPSILON="$1"
            elif [[ -z "$STEP_SIZE" ]]; then
                STEP_SIZE="$1"
            else
                NUM_STEPS="$1"
            fi
            shift ;;
    esac
done

case "$MODE" in
    all|ablation|experiments|subsampling|additional|additional_generators|dataset-generation) ;;
    *) echo "Unknown --mode: $MODE (expected all|ablation|experiments|subsampling|additional|additional_generators|dataset-generation)" >&2; exit 1 ;;
esac

# ──────────────── Required environment (see env_set.sh.example) ────────────────
: "${TEST_PATH:?Set TEST_PATH to the ELSA_TEST dataset directory (see env_set.sh)}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the directory where adversarial datasets are written (see env_set.sh)}"

# ──────────────── Optional environment (defaults shown) ────────────────
: "${BATCH_SIZE:=4}"
: "${WHITEBOX_SUBSET:=1000}"
: "${LOO_SUBSET:=1000}"
: "${RAID_SUBSET:=-1}"   # -1 = full dataset (~24k samples)

MODELS=(ojha2023 corvi2023 cavia2024 chen2024_convnext chen2024_clip koutlis2024 wang2020)

# External generators in $ADDGEN_SRC not already in ELSA_TEST (excluded:
# deepfloyd_if, compvis_stable_diffusion_v1_4, stabilityai_stable_diffusion_2_1_base,
# stabilityai_stable_diffusion_xl_base_1_0).
ADD_GENERATORS=(amused kandisky_2_1 kandisky_2_2 kandisky_3
                latent_consistency_model_simianluo latent_consistency_xl
                multi_diffusion pixart_alpha sdxl_turbo
                self_attention_guidance stable_unclip)

# ─────────────────────────── shared helpers ──────────────────────────────────

build_models_str() {
    local arr=("$@")
    local s="['${arr[0]}'"
    for det in "${arr[@]:1}"; do
        s+=", '${det}'"
    done
    s+="]"
    echo "$s"
}

# Runs one attack_generate.py call. All models in the ensemble share $DEVICE.
attack() {
    local subset="$1"; shift
    local models=("$@")
    local devices=()
    for _ in "${models[@]}"; do devices+=("$DEVICE"); done

    python raid/attack_generate.py \
        --models "${models[@]}" \
        --device "${devices[@]}" \
        --path_to_dataset "$TEST_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --epsilon "$EPSILON" \
        --num_steps "$NUM_STEPS" \
        --step_size "$STEP_SIZE" \
        --subset "$subset" \
        --generate
}

# ─────────────────────────── mode: ablation ──────────────────────────────────

run_ablation() {
    local epsilon="${EPSILON:?--epsilon is required for --mode ablation}"

    local step_counts=(10 20 30)
    local step_sizes=(0.01 0.03 0.05)
    local seeds=(42 234 123)
    local subset=200

    local total_runs=$(( ${#step_counts[@]} * ${#step_sizes[@]} * ${#seeds[@]} * ${#MODELS[@]} ))

    echo "============================================"
    echo "PGD Ablation Sweep"
    echo "Device:     $DEVICE"
    echo "Epsilon:    $epsilon"
    echo "Steps:      ${step_counts[*]}"
    echo "Step sizes: ${step_sizes[*]}"
    echo "Seeds:      ${seeds[*]}"
    echo "Subset:     $subset"
    echo "Total runs: $total_runs"
    echo "============================================"

    local eps_str="${epsilon//\//_}"
    local completed=0 skipped=0
    local start_time=$(date +%s)

    for seed in "${seeds[@]}"; do
        local seed_dir="${OUTPUT_DIR}_seed${seed}"
        for steps in "${step_counts[@]}"; do
            for alpha in "${step_sizes[@]}"; do
                for held_out in "${MODELS[@]}"; do
                    local ensemble=()
                    for det in "${MODELS[@]}"; do
                        [ "$det" != "$held_out" ] && ensemble+=("$det")
                    done

                    completed=$((completed + 1))

                    local models_str; models_str=$(build_models_str "${ensemble[@]}")
                    local expected_dir="${seed_dir}/ADV/ELSA_TEST_${subset}/adv_raw_${models_str}_${eps_str}_${steps}_${alpha}"
                    if [ -f "${expected_dir}/adv_dataset.pkl" ]; then
                        skipped=$((skipped + 1))
                        echo "[$completed/$total_runs] SKIP (exists): steps=$steps alpha=$alpha seed=$seed held_out=$held_out"
                        continue
                    fi

                    local devices=()
                    for _ in "${ensemble[@]}"; do devices+=("$DEVICE"); done

                    echo ""
                    echo "── [$completed/$total_runs] seed=$seed steps=$steps alpha=$alpha held_out=$held_out ──"

                    python raid/attack_generate.py \
                        --random_start 1 \
                        --random_seed "$seed" \
                        --models "${ensemble[@]}" \
                        --device "${devices[@]}" \
                        --path_to_dataset "$TEST_PATH" \
                        --output_dir "$seed_dir" \
                        --batch_size "$BATCH_SIZE" \
                        --epsilon "$epsilon" \
                        --num_steps "$steps" \
                        --step_size "$alpha" \
                        --subset "$subset" \
                        --generate
                done
            done
        done
    done

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo ""
    echo "============================================"
    echo "Ablation complete: $completed runs ($skipped skipped) in $(printf '%dh %dm %ds' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))"
    echo "============================================"
}

# ─────────────────────────── mode: experiments ───────────────────────────────

run_experiments() {
    : "${EPSILON:?--epsilon is required for --mode experiments}"
    : "${STEP_SIZE:?--step_size is required for --mode experiments}"

    echo "============================================"
    echo "RAID experiment run"
    echo "TEST_PATH:   $TEST_PATH"
    echo "OUTPUT_DIR:  $OUTPUT_DIR"
    echo "DEVICE:      $DEVICE"
    echo "EPSILON:     $EPSILON   STEP_SIZE: $STEP_SIZE   NUM_STEPS: $NUM_STEPS"
    echo "============================================"

    echo ""
    echo "── 1. White-box attacks (subset=$WHITEBOX_SUBSET) ──"
    for m in "${MODELS[@]}"; do
        echo "  -> $m"
        attack "$WHITEBOX_SUBSET" "$m"
    done

    echo ""
    echo "── 2. Leave-one-out ensemble attacks (subset=$LOO_SUBSET) ──"
    for held_out in "${MODELS[@]}"; do
        ensemble=()
        for m in "${MODELS[@]}"; do
            [ "$m" != "$held_out" ] && ensemble+=("$m")
        done
        echo "  -> held out: $held_out"
        attack "$LOO_SUBSET" "${ensemble[@]}"
    done

    echo ""
    echo "── 3. Full RAID ensemble attack (subset=$RAID_SUBSET) ──"
    attack "$RAID_SUBSET" "${MODELS[@]}"

    echo ""
    echo "Done."
}

# ─────────────────────────── mode: subsampling ──────────────────────────────

run_subsampling() {
    : "${EPSILON:?--epsilon is required for --mode subsampling}"
    : "${STEP_SIZE:?--step_size is required for --mode subsampling}"

    local seeds
    read -ra seeds <<< "$SUB_SEEDS"
    local subset="$SUB_SUBSET"
    local eps_str="${EPSILON//\//_}"
    local total_runs=$(( ${#seeds[@]} * (${#MODELS[@]} + ${#MODELS[@]}) ))

    echo "============================================"
    echo "Subsampling (5-seed variance estimation)"
    echo "Device:     $DEVICE"
    echo "Epsilon:    $EPSILON   Step size: $STEP_SIZE   Steps: $NUM_STEPS"
    echo "Seeds:      ${seeds[*]}"
    echo "Subset:     $subset"
    echo "Total runs: $total_runs (${#MODELS[@]} white-box + ${#MODELS[@]} LOO) x ${#seeds[@]} seeds"
    echo "============================================"

    local completed=0 skipped=0
    local start_time=$(date +%s)

    for seed in "${seeds[@]}"; do
        local split_dir="${OUTPUT_DIR}_mean/${seed}"

        echo ""
        echo "══════ seed=$seed  output=$split_dir ══════"

        # ── White-box (single-model) attacks ──
        for m in "${MODELS[@]}"; do
            completed=$((completed + 1))

            local expected="${split_dir}/ADV/ELSA_TEST_${subset}/adv_raw_${m}_${eps_str}_${NUM_STEPS}_${STEP_SIZE}"
            if [ -f "${expected}/adv_dataset.pkl" ]; then
                skipped=$((skipped + 1))
                echo "[$completed/$total_runs] SKIP white-box $m seed=$seed"
                continue
            fi

            echo "[$completed/$total_runs] white-box $m seed=$seed"
            python raid/attack_generate.py \
                --random_start 1 \
                --random_seed "$seed" \
                --sample_seed "$seed" \
                --models "$m" \
                --device "$DEVICE" \
                --path_to_dataset "$TEST_PATH" \
                --output_dir "$split_dir" \
                --batch_size "$BATCH_SIZE" \
                --epsilon "$EPSILON" \
                --num_steps "$NUM_STEPS" \
                --step_size "$STEP_SIZE" \
                --subset "$subset" \
                --generate
        done

        # ── Leave-one-out ensemble attacks ──
        for held_out in "${MODELS[@]}"; do
            local ensemble=()
            for m in "${MODELS[@]}"; do
                [ "$m" != "$held_out" ] && ensemble+=("$m")
            done

            completed=$((completed + 1))

            local models_str; models_str=$(build_models_str "${ensemble[@]}")
            local expected="${split_dir}/ADV/ELSA_TEST_${subset}/adv_raw_${models_str}_${eps_str}_${NUM_STEPS}_${STEP_SIZE}"
            if [ -f "${expected}/adv_dataset.pkl" ]; then
                skipped=$((skipped + 1))
                echo "[$completed/$total_runs] SKIP LOO held_out=$held_out seed=$seed"
                continue
            fi

            local devices=()
            for _ in "${ensemble[@]}"; do devices+=("$DEVICE"); done

            echo "[$completed/$total_runs] LOO held_out=$held_out seed=$seed"
            python raid/attack_generate.py \
                --random_start 1 \
                --random_seed "$seed" \
                --sample_seed "$seed" \
                --models "${ensemble[@]}" \
                --device "${devices[@]}" \
                --path_to_dataset "$TEST_PATH" \
                --output_dir "$split_dir" \
                --batch_size "$BATCH_SIZE" \
                --epsilon "$EPSILON" \
                --num_steps "$NUM_STEPS" \
                --step_size "$STEP_SIZE" \
                --subset "$subset" \
                --generate
        done
    done

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo ""
    echo "============================================"
    echo "Subsampling complete: $completed runs ($skipped skipped) in $(printf '%dh %dm %ds' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))"
    echo "============================================"
}

# ─────────────────────────── mode: additional ────────────────────────────────

run_additional() {
    : "${EPSILON:?--epsilon is required for --mode additional}"
    : "${STEP_SIZE:?--step_size is required for --mode additional}"

    # Same fixed epsilon/num_steps/step_size as PGD, same LOO subset (1000) -
    # a like-for-like comparison, no attack-specific hyperparameter tuning.
    local attacks=(apgd cwa)
    local subset="$LOO_SUBSET"
    local eps_str="${EPSILON//\//_}"
    local total_runs=$(( ${#attacks[@]} * ${#MODELS[@]} ))

    echo "============================================"
    echo "Additional attacks (APGD, CWA) - leave-one-out"
    echo "Device:     $DEVICE"
    echo "Epsilon:    $EPSILON   Step size: $STEP_SIZE   Steps: $NUM_STEPS"
    echo "Subset:     $subset"
    echo "Attacks:    ${attacks[*]}"
    echo "Total runs: $total_runs"
    echo "============================================"

    local completed=0 skipped=0
    local start_time=$(date +%s)

    for attack_type in "${attacks[@]}"; do
        for held_out in "${MODELS[@]}"; do
            local ensemble=()
            for m in "${MODELS[@]}"; do
                [ "$m" != "$held_out" ] && ensemble+=("$m")
            done

            completed=$((completed + 1))

            local models_str; models_str=$(build_models_str "${ensemble[@]}")
            local expected="${OUTPUT_DIR}/ADV/ELSA_TEST_${subset}/adv_${attack_type}_${models_str}_${eps_str}_${NUM_STEPS}_${STEP_SIZE}"
            if [ -f "${expected}/adv_dataset.pkl" ]; then
                skipped=$((skipped + 1))
                echo "[$completed/$total_runs] SKIP attack=$attack_type held_out=$held_out"
                continue
            fi

            local devices=()
            for _ in "${ensemble[@]}"; do devices+=("$DEVICE"); done

            echo ""
            echo "── [$completed/$total_runs] attack=$attack_type held_out=$held_out ──"

            python raid/attack_generate.py \
                --attack "$attack_type" \
                --models "${ensemble[@]}" \
                --device "${devices[@]}" \
                --path_to_dataset "$TEST_PATH" \
                --output_dir "$OUTPUT_DIR" \
                --batch_size "$BATCH_SIZE" \
                --epsilon "$EPSILON" \
                --num_steps "$NUM_STEPS" \
                --step_size "$STEP_SIZE" \
                --subset "$subset" \
                --generate
        done
    done

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo ""
    echo "============================================"
    echo "Additional attacks complete: $completed runs ($skipped skipped) in $(printf '%dh %dm %ds' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))"
    echo "============================================"
}

# ────────────────────── mode: additional_generators ─────────────────────────

run_additional_generators() {
    : "${EPSILON:?--epsilon is required for --mode additional_generators}"
    : "${STEP_SIZE:?--step_size is required for --mode additional_generators}"
    : "${ADDGEN_SRC:?Set ADDGEN_SRC to the external generators directory (see env_set.sh)}"
    : "${ADDGEN_PATH:=$(dirname "$TEST_PATH")/EXTERNAL_GEN}"

    # Dataset dir of symlinks: one subfolder per generator + real/ from
    # ELSA_TEST, so get_dataloader's "subfolders" layout and the ADV/<name>_<subset>/
    # output naming work unchanged. Never name it "modern_generator" - the
    # loader skips folders whose path contains that.
    mkdir -p "$ADDGEN_PATH"
    ln -sfn "$TEST_PATH/real" "$ADDGEN_PATH/real"
    for g in "${ADD_GENERATORS[@]}"; do
        [ -d "$ADDGEN_SRC/$g" ] || { echo "Missing generator dir: $ADDGEN_SRC/$g" >&2; exit 1; }
        ln -sfn "$ADDGEN_SRC/$g" "$ADDGEN_PATH/$g"
    done

    local subset="$SUB_SUBSET"
    local eps_str="${EPSILON//\//_}"
    local adv_root="${OUTPUT_DIR}/ADV/$(basename "$ADDGEN_PATH")_${subset}"

    # (label, models...) jobs, in the order given by --gen_attacks
    local jobs=()
    for a in $GEN_ATTACKS; do
        case "$a" in
            wb)   for m in "${MODELS[@]}"; do jobs+=("wb:$m"); done ;;
            loo)  for m in "${MODELS[@]}"; do jobs+=("loo:$m"); done ;;
            *) echo "Unknown --gen_attacks entry: $a (expected wb|loo)" >&2; exit 1 ;;
        esac
    done
    local total_runs=${#jobs[@]}

    echo "============================================"
    echo "Additional generators"
    echo "Device:     $DEVICE"
    echo "Epsilon:    $EPSILON   Step size: $STEP_SIZE   Steps: $NUM_STEPS"
    echo "Dataset:    $ADDGEN_PATH (${#ADD_GENERATORS[@]} generators + real)"
    echo "Subset:     $subset per folder"
    echo "Attacks:    $GEN_ATTACKS"
    echo "Total runs: $total_runs"
    echo "============================================"

    local completed=0 skipped=0
    local start_time=$(date +%s)

    for job in "${jobs[@]}"; do
        local kind="${job%%:*}" arg="${job#*:}"
        local ensemble=()
        case "$kind" in
            wb)   ensemble=("$arg") ;;
            loo)  for m in "${MODELS[@]}"; do [ "$m" != "$arg" ] && ensemble+=("$m"); done ;;
        esac

        completed=$((completed + 1))

        local models_str; models_str=$(build_models_str "${ensemble[@]}")
        local expected="${adv_root}/adv_raw_${models_str}_${eps_str}_${NUM_STEPS}_${STEP_SIZE}"
        if [ -f "${expected}/adv_dataset.pkl" ]; then
            skipped=$((skipped + 1))
            echo "[$completed/$total_runs] SKIP $kind $arg"
            continue
        fi

        local devices=()
        for _ in "${ensemble[@]}"; do devices+=("$DEVICE"); done

        echo ""
        echo "── [$completed/$total_runs] $kind $arg ──"

        python raid/attack_generate.py \
            --models "${ensemble[@]}" \
            --device "${devices[@]}" \
            --path_to_dataset "$ADDGEN_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --epsilon "$EPSILON" \
            --num_steps "$NUM_STEPS" \
            --step_size "$STEP_SIZE" \
            --subset "$subset" \
            --generate
    done

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo ""
    echo "============================================"
    echo "Additional generators complete: $completed runs ($skipped skipped) in $(printf '%dh %dm %ds' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))"
    echo "============================================"
}

# ────────────────────── mode: dataset-generation ───────────────────────────

run_dataset_generation() {
    : "${EPSILON:?--epsilon is required for --mode dataset-generation}"
    : "${STEP_SIZE:?--step_size is required for --mode dataset-generation}"

    local eps_str="${EPSILON//\//_}"
    local models_str; models_str=$(build_models_str "${MODELS[@]}")
    local name_prefix
    if [[ "$ATTACK" == "pgd" ]]; then
        name_prefix="raw"
    else
        name_prefix="$ATTACK"
    fi
    local expected="${OUTPUT_DIR}/ADV/ELSA_TEST/adv_${name_prefix}_${models_str}_${eps_str}_${NUM_STEPS}_${STEP_SIZE}"

    echo "============================================"
    echo "Dataset generation (full RAID ensemble)"
    echo "Device:     $DEVICE"
    echo "Attack:     $ATTACK"
    echo "Epsilon:    $EPSILON   Step size: $STEP_SIZE   Steps: $NUM_STEPS"
    echo "Subset:     $RAID_SUBSET (full dataset)"
    echo "Expected:   $expected"
    echo "============================================"

    if [ -f "${expected}/adv_dataset.pkl" ]; then
        echo "SKIP: ${expected}/adv_dataset.pkl already exists"
        return
    fi

    local devices=()
    for _ in "${MODELS[@]}"; do devices+=("$DEVICE"); done

    python raid/attack_generate.py \
        --attack "$ATTACK" \
        --models "${MODELS[@]}" \
        --device "${devices[@]}" \
        --path_to_dataset "$TEST_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --epsilon "$EPSILON" \
        --num_steps "$NUM_STEPS" \
        --step_size "$STEP_SIZE" \
        --subset "$RAID_SUBSET" \
        --generate

    echo "Done."
}

# ──────────────────────────────── dispatch ───────────────────────────────────

case "$MODE" in
    ablation)
        run_ablation
        ;;
    experiments)
        run_experiments
        ;;
    subsampling)
        run_subsampling
        ;;
    additional)
        run_additional
        ;;
    additional_generators)
        run_additional_generators
        ;;
    dataset-generation)
        run_dataset_generation
        ;;
    all)
        run_ablation
        run_experiments
        run_subsampling
        run_additional
        ;;
esac
