#!/usr/bin/env python3
"""Builds the paper's result tables from evaluate_detector.py output.

  python build_results_table.py [8 16 32]                     # both, ELSA then external
  python build_results_table.py [8 16 32] --dataset elsa      # ELSA_TEST_1000 only
  python build_results_table.py [8 16 32] --dataset external  # EXTERNAL_GEN_200 only
                                                              # (11 external gens + ELSA reals)

Tables (rows/cols in ORDER; row = attack source, col = eval detector):

  --dataset elsa      (adv: output/ADV/ELSA_TEST_1000, clean: clean/ELSA_1000)
    1. Clean performance            F1 / Acc / AUROC per detector
    2. AUROC_robust / ASR%, per eps white-box rows (7x7) + N-model (LOO) row
    3. LOO ASR% by attack, per eps  PGD / APGD / CWA rows
    4. F1, per eps                  white-box rows (7x7) + N-model (LOO) row

  --dataset external  (adv: output/ADV/EXTERNAL_GEN_200, clean: clean/EXTERNAL_GEN_200)
    1. Clean performance            as above, on the external generators
    2. AUROC_robust / ASR%, per eps as above
    4. F1, per eps                  as above
    (no table 3: APGD/CWA were not run on the external generators)

  --dataset elsa only:
    5. RAID dataset             adv: output/ADV/ELSA_TEST (full, 7-model ensemble),
                                clean: clean/ELSA_full; rows = 7 ensemble detectors +
                                evaluation-only ones (DINOv2, ViT-T, Effort, AIDE, D3, OmniAID, DDA)
    6. RAID perceptual          LPIPS / SSIM mean ± std over images of the RAID dataset
                                (PGD, 7-model ensemble) vs clean, per eps
                                (compute_metrics.py --mode raid_perceptual); falls back to
                                the pooled mean over subset folders of the same config

ASR = (clean_AUC - adv_AUC) / clean_AUC * 100, clean AUC from the matching
clean baseline dir.
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Paper's row/column order (display name -> internal model name)
ORDER = [
    ("Ca24", "cavia2024"),
    ("Ch24C", "chen2024_clip"),
    ("Ch24CN", "chen2024_convnext"),
    ("Co23", "corvi2023"),
    ("K24", "koutlis2024"),
    ("O23", "ojha2023"),
    ("W20", "wang2020"),
]
MODELS = [m for _, m in ORDER]

# Evaluation-only detectors (not in any attack ensemble), Table 5 rows only.
NEW_ORDER = [
    ("DINOv2", "vit_lp14_dinov2"),
    ("DINOv2reg", "vit_lp14_reg_dinov2"),
    ("ViT-T", "vit_tp16_224_augreg_in21k"),
    ("ViT-Tcode", "vit_tp16_224_code_augreg_in21k"),
    ("Effort", "effort"),
    ("AIDE", "aide"),
    ("D3", "d3"),
    ("OmniAID", "omniaid"),
    ("DDA", "dda"),
]

# Order used by run_leave_one_out_attacks.sh's DETECTORS array when building
# the ensemble (and thus the folder name) -- must match exactly, independent
# of the paper's display order above.
ATTACK_ORDER = ["ojha2023", "corvi2023", "cavia2024", "chen2024_convnext",
                "chen2024_clip", "koutlis2024", "wang2020"]

# Step sizes actually used per epsilon budget (must match run_experiments.sh).
STEP_PARAMS = {
    "8":  "10_0.01",
    "16": "10_0.03",
    "32": "10_0.03",
}

RAID_ROOT = os.path.dirname(SRC_DIR)
EVAL_BASE = os.environ.get(
    "EVAL_OUTPUT_DIR",
    os.path.join(RAID_ROOT, "results/evaluate_detector"),
)
# (attacked results dir, clean baseline dir) per dataset; --dataset picks one
DATASETS = {
    "elsa":     ("output/ADV/ELSA_TEST_1000", "clean/ELSA_1000"),
    "external": ("output/ADV/EXTERNAL_GEN_200", "clean/EXTERNAL_GEN_200"),
}
DATASET_DESC = {
    "elsa":     "ELSA_TEST, 1000 images per folder (4 generators + real)",
    "external": "EXTERNAL_GEN, 200 images per folder (11 external generators + ELSA real)",
}
BASE = os.path.join(EVAL_BASE, DATASETS["elsa"][0])
CLEAN_BASE = os.path.join(EVAL_BASE, DATASETS["elsa"][1])


def model_list_repr(models):
    return "[" + ", ".join(f"'{m}'" for m in models) + "]"


def load_json_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["metrics"]["All"]


def load_metrics(source_models, eval_model, eps, attack="raw"):
    """metrics['All'] for an attack source evaluated by eval_model."""
    src_repr = model_list_repr(source_models)
    params = STEP_PARAMS.get(eps, "10_0.03")
    return load_json_metrics(
        f"{BASE}/adv_{attack}_{src_repr}_{eps}_255_{params}/{eval_model}/results.json")


def load_clean(eval_model):
    return load_json_metrics(f"{CLEAN_BASE}/{eval_model}/results.json")


# ──────────────────────────── cell formatters ─────────────────────────────────

def cell_auc_asr(adv, clean):
    """AUROC_robust / ASR%"""
    if adv is None or clean is None:
        return "---"
    asr = (clean['auc'] - adv['auc']) / clean['auc'] * 100
    return f"{adv['auc']:.2f}/{asr:.1f}"


def cell_asr(adv, clean):
    """ASR = (clean_AUC - adv_AUC) / clean_AUC * 100"""
    if adv is None or clean is None:
        return "---"
    return f"{(clean['auc'] - adv['auc']) / clean['auc'] * 100:.1f}"


def cell_f1(adv, clean):
    if adv is None:
        return "---"
    return f"{adv['f1']:.2f}"


# ──────────────────────────── table builders ──────────────────────────────────

def build_table(eps, cell_fn, attack="raw"):
    rows = []

    # white-box rows: source = [row_model], evaluated against every column
    for disp, row_model in ORDER:
        row = [disp]
        for _, col_model in ORDER:
            row.append(cell_fn(load_metrics([row_model], col_model, eps, attack),
                               load_clean(col_model)))
        rows.append(row)

    # N-model row: for each column, source = all models except that column,
    # evaluated by that same column model (the held-out one)
    row = ["N-model"]
    for _, col_model in ORDER:
        ensemble = [m for m in ATTACK_ORDER if m != col_model]
        row.append(cell_fn(load_metrics(ensemble, col_model, eps, attack),
                           load_clean(col_model)))
    rows.append(row)

    return rows


def build_additional_table(eps, cell_fn):
    """LOO results comparing PGD vs APGD vs CWA.
    Rows = attack method, columns = held-out detector (evaluated on itself)."""
    rows = []
    for attack_name, attack_prefix in [("PGD", "raw"), ("APGD", "apgd"), ("CWA", "cwa")]:
        row = [attack_name]
        for _, col_model in ORDER:
            ensemble = [m for m in ATTACK_ORDER if m != col_model]
            row.append(cell_fn(load_metrics(ensemble, col_model, eps, attack_prefix),
                               load_clean(col_model)))
        rows.append(row)
    return rows


# ──────────────────────────── rendering ───────────────────────────────────────

def render(header, rows, title):
    widths = [max(len(str(r[i])) for r in ([header] + rows))
              for i in range(len(header))]

    def fmt_row(r):
        return "  ".join(str(c).ljust(w) for c, w in zip(r, widths))

    print(f"\n=== {title} ===")
    print(fmt_row(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))


def print_raid_table(epsilons):
    """Table 5: the RAID dataset (full ELSA_TEST, 7-model ensemble) evaluated
    by every detector - the 7 ensemble members and the evaluation-only ones.
    Baseline = clean/ELSA_full (compute_metrics.py --mode clean --subset -1)."""
    base = os.path.join(EVAL_BASE, "output/ADV/ELSA_TEST")
    clean_base = os.path.join(EVAL_BASE, "clean/ELSA_full")
    attacks = [("PGD", "raw"), ("APGD", "apgd")]
    header = ["Detector"] + [f"{a} {e}/255" for e in epsilons for a, _ in attacks]
    src = model_list_repr(ATTACK_ORDER)
    rows = []
    for disp, model in ORDER + NEW_ORDER:
        clean = load_json_metrics(f"{clean_base}/{model}/results.json")
        row = [disp + ("" if model in MODELS else " *")]
        for eps in epsilons:
            params = STEP_PARAMS.get(eps, "10_0.03")
            for _, prefix in attacks:
                adv = load_json_metrics(
                    f"{base}/adv_{prefix}_{src}_{eps}_255_{params}/{model}/results.json")
                row.append(cell_auc_asr(adv, clean))
        rows.append(row)
    render(header, rows,
           "[elsa] Table 5 -- RAID dataset (full ELSA_TEST, 7-model ensemble)  "
           "AUROC_robust / ASR%  (* = not in the ensemble)")

    # Same, on the PNG-quantized release (compute_metrics.py --mode raid_img)
    rows = []
    for disp, model in ORDER + NEW_ORDER:
        clean = load_json_metrics(f"{clean_base}/{model}/results.json")
        row = [disp + ("" if model in MODELS else " *")]
        for eps in epsilons:
            params = STEP_PARAMS.get(eps, "10_0.03")
            for _, prefix in attacks:
                adv = load_json_metrics(
                    f"{base}/adv_{prefix}_{src}_{eps}_255_{params}/IMAGE_DATASET/{model}/results.json")
                row.append(cell_auc_asr(adv, clean))
        rows.append(row)
    render(header, rows,
           "[elsa] Table 5 (RAID Img) -- PNG release of the RAID dataset  "
           "AUROC_robust / ASR%  (* = not in the ensemble)")


def load_sidecar(path, key):
    """(mean, std) from a compute_metrics.py lpips/ssim sidecar, or None."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d[key], d[f"{key}_std"]


def cell_mean_std(v):
    return "---" if v is None else f"{v[0]:.3f} ± {v[1]:.3f}"


def pool_mean_std(vals):
    """Pooled (mean, std) over sidecars with equal n per sidecar:
    var = E[std_i^2 + mean_i^2] - mean^2."""
    mean = sum(mu for mu, _ in vals) / len(vals)
    second = sum(sd ** 2 + mu ** 2 for mu, sd in vals) / len(vals)
    return mean, math.sqrt(max(second - mean ** 2, 0.0))


def load_raid_perceptual(eps, attack="raw", lpips_net="alex"):
    """{"lpips": (mean, std), "ssim": ...} of the RAID dataset (7-model
    ensemble, selected STEP_PARAMS) vs clean, and the source used.
    Full RAID sidecars (compute_metrics.py --mode raid_perceptual) if present,
    otherwise pooled over every subset folder of the same attack config."""
    folder = f"adv_{attack}_{model_list_repr(ATTACK_ORDER)}_{eps}_255_{STEP_PARAMS.get(eps, '10_0.03')}"
    full = os.path.join(EVAL_BASE, "output/ADV/ELSA_TEST", folder)
    out, source = {}, {}
    for metric, fname in (("lpips", f"lpips_{lpips_net}.json"), ("ssim", "ssim.json")):
        v = load_sidecar(os.path.join(full, fname), metric)
        if v is not None:
            out[metric], source[metric] = v, "full RAID"
            continue
        vals = [load_sidecar(f, metric) for f in glob.glob(
            os.path.join(glob.escape(EVAL_BASE), "**", glob.escape(folder), fname), recursive=True)]
        vals = [v for v in vals if v is not None]
        if vals:
            out[metric], source[metric] = pool_mean_std(vals), f"mean of {len(vals)} subsets"
    return out, source


def print_raid_perceptual_table(epsilons, lpips_net="alex"):
    """Table 6: LPIPS / SSIM of the RAID dataset (PGD, 7-model ensemble) vs
    clean images, mean ± std over images, one row per eps."""
    rows, notes = [], []
    for eps in epsilons:
        data, source = load_raid_perceptual(eps, lpips_net=lpips_net)
        rows.append([f"{eps}/255", cell_mean_std(data.get("lpips")),
                     cell_mean_std(data.get("ssim"))])
        for metric, src in source.items():
            if src != "full RAID":
                notes.append(f"{eps}/255 {metric}: {src} (full RAID sidecar missing)")
    render(["eps", f"LPIPS ({lpips_net})", "SSIM"], rows,
           "[elsa] Table 6 -- RAID dataset (PGD, 7-model ensemble) LPIPS / SSIM vs clean, "
           "mean ± std over images")
    for msg in notes:
        print(f"  NOTE: {msg}")


SEED_DIRS = "output_seed*"


def load_loo_seed_metrics(eps, held_out):
    """AUROC of the leave-one-out PGD attack (selected STEP_PARAMS) on the
    held-out detector, one value per random-start seed of the PGD ablation
    (output_seed*/ADV/ELSA_TEST_200, 1,000 images)."""
    src = model_list_repr([m for m in ATTACK_ORDER if m != held_out])
    folder = f"adv_raw_{src}_{eps}_255_{STEP_PARAMS.get(eps, '10_0.03')}"
    pattern = os.path.join(glob.escape(EVAL_BASE), SEED_DIRS, "ADV", "ELSA_TEST_200",
                           glob.escape(folder), held_out, "results.json")
    return {re.search(r"output_seed(\d+)", f).group(1): load_json_metrics(f)["auc"]
            for f in sorted(glob.glob(pattern))}


def print_seed_variability_table(epsilons):
    """Table 7: leave-one-out AUROC / ASR% on each held-out detector, mean ±
    std over the random-start seeds of the PGD ablation. ASR per seed w.r.t.
    the clean AUROC on the same 1,000 images (clean/ELSA_1000)."""
    clean_base = os.path.join(EVAL_BASE, "clean/ELSA_1000")
    header = ["eps"] + [disp for disp, _ in ORDER]
    rows, seeds_seen = [], set()
    for eps in epsilons:
        auc_row, asr_row = [f"{eps}/255 AUROC"], [f"{eps}/255 ASR%"]
        for _, model in ORDER:
            vals = load_loo_seed_metrics(eps, model)
            clean = load_json_metrics(f"{clean_base}/{model}/results.json")
            seeds_seen |= set(vals)
            if not vals or clean is None:
                auc_row.append("---"); asr_row.append("---")
                continue
            aucs = list(vals.values())
            asrs = [100 * (clean["auc"] - a) / clean["auc"] for a in aucs]
            sd = lambda v: math.sqrt(sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v))
            auc_row.append(f"{sum(aucs)/len(aucs):.2f} ± {sd(aucs):.2f} (n={len(aucs)})")
            asr_row.append(f"{sum(asrs)/len(asrs):.1f} ± {sd(asrs):.1f}")
        rows += [auc_row, asr_row]
    render(header, rows,
           f"[elsa] Table 7 -- LOO PGD, mean ± std over random-start seeds "
           f"{sorted(seeds_seen)} (PGD ablation, 1,000 images)")


def auc_first_n(results_json, n):
    """AUROC of a results.json restricted to the first n images of every class
    block, recomputed from its stored logits (blocks are stored in the order of
    metrics' keys, equal size). Matches the unseeded prefix subsets (--subset n)
    of attack_generate.py; None if the file is missing."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    if not os.path.exists(results_json):
        return None
    with open(results_json) as f:
        d = json.load(f)
    keys = [k for k in d["metrics"] if k != "All"]
    logits = np.asarray(d["logits"], dtype=np.float64)
    per = len(logits) // len(keys)
    if n > per:
        return None
    z = logits - logits.max(1, keepdims=True)
    prob = np.exp(z[:, 1]) / np.exp(z).sum(1)
    idx, y = [], []
    for i, k in enumerate(keys):
        idx += range(i * per, i * per + n)
        y += [0 if k == "real" else 1] * n
    return roc_auc_score(y, prob[idx])


def print_other_attacks_table(epsilons, n=150):
    """Table 3b: LOO ASR% of PGD / APGD / CWA on the same first-n-per-class
    images. PGD/APGD from the ELSA_TEST_1000 results (logits sliced), CWA from
    its own ELSA_TEST_<n> run; clean AUROC from clean/ELSA_1000, sliced alike."""
    header = ["eps", "Attack"] + [d for d, _ in ORDER] + ["Mean"]
    rows = []
    for eps in epsilons:
        params = STEP_PARAMS.get(eps, "10_0.03")
        for disp, attack, ds in (("PGD", "raw", "ELSA_TEST_1000"), ("APGD", "apgd", "ELSA_TEST_1000"),
                                 ("CWA", "cwa", f"ELSA_TEST_{n}")):
            row, vals = [f"{eps}/255", disp], []
            for _, model in ORDER:
                src = model_list_repr([m for m in ATTACK_ORDER if m != model])
                adv = auc_first_n(os.path.join(EVAL_BASE, "output/ADV", ds,
                                  f"adv_{attack}_{src}_{eps}_255_{params}", model, "results.json"), n)
                clean = auc_first_n(os.path.join(EVAL_BASE, "clean/ELSA_1000", model, "results.json"), n)
                if adv is None or clean is None:
                    row.append("---")
                    continue
                vals.append(100 * (clean - adv) / clean)
                row.append(f"{vals[-1]:.1f}")
            row.append(f"{sum(vals) / len(vals):.1f}" if len(vals) == len(ORDER) else "---")
            rows.append(row)
    render(header, rows, f"[elsa] Table 3b -- LOO ASR% by attack on the same {n} images per class")


TRANSFORM_ROWS = [("No transform", None), ("Resize 64x64", "resize64"),
                  ("Random crop 64x64", "crop64"), ("Random flip", "flip"), ("JPEG (q=75)", "jpeg75")]


def print_transforms_table(epsilons):
    """Table C.10: held-out AUROC / ASR% of the leave-one-out PGD attack
    (ELSA_TEST_1000) with a post-processing applied to the adversarial images
    only; ASR w.r.t. the untransformed clean AUROC (clean/ELSA_1000)."""
    base = os.path.join(EVAL_BASE, "output/ADV/ELSA_TEST_1000")
    clean_base = os.path.join(EVAL_BASE, "clean/ELSA_1000")
    for eps in epsilons:
        rows = []
        for disp, t in TRANSFORM_ROWS:
            row = [disp]
            for _, model in ORDER:
                src = model_list_repr([m for m in ATTACK_ORDER if m != model])
                folder = f"{base}/adv_raw_{src}_{eps}_255_{STEP_PARAMS.get(eps, '10_0.03')}"
                path = (f"{folder}/{model}/results.json" if t is None
                        else f"{folder}/transforms/{t}/{model}/results.json")
                row.append(cell_auc_asr(load_json_metrics(path),
                                        load_json_metrics(f"{clean_base}/{model}/results.json")))
            rows.append(row)
        render(["Transform"] + [d for d, _ in ORDER], rows,
               f"[elsa] Table C.10 -- eps={eps}/255 LOO PGD + post-processing on the adversarial "
               f"images: AUROC / ASR% (vs untransformed clean)")


def print_clean_table():
    rows = []
    for disp, model in ORDER:
        m = load_clean(model)
        if m is None:
            rows.append([disp, "---", "---", "---"])
        else:
            rows.append([disp, f"{m['f1']:.2f}", f"{m['acc']:.2f}",
                         f"{m['auc']:.2f}"])
    render(["Detector", "F1", "Acc", "AUROC"], rows,
           f"[{TAG}] Table 1 -- clean performance")


def print_dataset(dataset, epsilons):
    """All tables for one dataset. BASE/CLEAN_BASE/TAG are module globals
    read by load_metrics/load_clean/print_clean_table."""
    global BASE, CLEAN_BASE, TAG
    BASE = os.path.join(EVAL_BASE, DATASETS[dataset][0])
    CLEAN_BASE = os.path.join(EVAL_BASE, DATASETS[dataset][1])
    TAG = dataset
    header = ["Attack \\ Eval"] + [disp for disp, _ in ORDER]

    print(f"\n{'#' * 80}")
    print(f"Dataset: {dataset} -- {DATASET_DESC[dataset]}")
    print(f"  adv results:    {BASE}")
    print(f"  clean baseline: {CLEAN_BASE}")
    print("Tables (row = attack source, col = eval detector; N-model = LOO ensemble"
          " evaluated on the held-out detector):")
    print("  1. Clean F1 / Acc / AUROC")
    print(f"  2. AUROC_robust / ASR%  per eps ({', '.join(epsilons)})")
    if dataset == "elsa":
        print("  3. LOO ASR% by attack (PGD / APGD / CWA)  per eps")
    else:
        print("  3. (skipped: APGD/CWA not run on this dataset)")
    print("  4. F1  per eps")
    if dataset == "elsa":
        print("  5. RAID dataset (full, 7-model ensemble) AUROC_robust / ASR% per detector,"
              " incl. evaluation-only detectors; baseline clean/ELSA_full")
        print("  6. RAID dataset LPIPS / SSIM vs clean (mean ± std over images) per eps")
        print("  7. LOO AUROC / ASR% mean ± std over the PGD-ablation random-start seeds")
    print("ASR = (clean_AUC - adv_AUC) / clean_AUC * 100")
    print('#' * 80)

    # 1. Clean performance
    print_clean_table()

    # 2. AUROC/ASR per epsilon (whitebox + LOO)
    for eps in epsilons:
        render(header, build_table(eps, cell_auc_asr),
               f"[{TAG}] Table 2 -- eps={eps}/255  AUROC_robust / ASR%")

    # 3. LOO ASR by attack method per epsilon (APGD/CWA only exist for ELSA)
    if dataset == "elsa":
        for eps in epsilons:
            render(header, build_additional_table(eps, cell_asr),
                   f"[{TAG}] Table 3 -- eps={eps}/255  LOO ASR% by attack method")

    # 4. F1 per epsilon (whitebox + LOO)
    for eps in epsilons:
        render(header, build_table(eps, cell_f1),
               f"[{TAG}] Table 4 -- eps={eps}/255  F1")

    # 5. Full RAID dataset, all detectors (ELSA only)
    if dataset == "elsa":
        print_raid_table(epsilons)
        print_raid_perceptual_table(epsilons)
        print_seed_variability_table(epsilons)
        print_other_attacks_table(epsilons)
        print_transforms_table(epsilons)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("epsilons", nargs="*", default=["8", "16", "32"])
    p.add_argument("--dataset", choices=["all"] + list(DATASETS), default="all",
                   help="all = ELSA then external")
    args = p.parse_args()
    for ds in (DATASETS if args.dataset == "all" else [args.dataset]):
        print_dataset(ds, args.epsilons)
