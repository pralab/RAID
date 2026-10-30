#!/usr/bin/env python3
"""Evaluate detectors and compute metrics, across three scenarios:

  clean       - all 7 detectors on the unattacked test set, plus the
                external generators ($ADDGEN_PATH) if that dataset exists
                (was run_clean_eval.sh)
  experiments - all 7 detectors on every white-box / leave-one-out / full-RAID
                adv_dataset.pkl under $OUTPUT_DIR/ADV/ (produced by
                run_experiments.sh's "experiments" mode)
                (was run_evaluate_all.sh)
  ablation    - AUROC + LPIPS + ASR for the PGD ablation grid (produced by
                run_experiments.sh's "ablation" mode) and the noise-control
                grid (produced by generate_noise.py --type control), then
                rebuilds ablation_metrics.csv
                (was compute_ablation_metrics.py)
  per_subset  - no GPU: per-generator AUROC/acc/ASR from the stored logits
                of every results.json under $EVAL_OUTPUT_DIR, written to
                results/per_subset_metrics.csv
  subsampling - all 7 detectors on every adv_dataset.pkl under
                $OUTPUT_DIR_mean/<seed>/ADV/ (produced by run_experiments.sh's
                "subsampling" mode). Results land in
                $EVAL_OUTPUT_DIR/mean/<seed>/ADV/.../<model>/results.json
  raid_perceptual - LPIPS + SSIM (mean ± std over images) of every full-RAID
                adv_dataset.pkl under $OUTPUT_DIR/ADV/<basename $TEST_PATH>/
                vs the clean full test set; sidecars next to its results
  raid_img    - detectors on the PNG release of the full RAID dataset
                (<adv dir>/IMAGE_DATASET/, written by
                raid/scripts/generate_image_dataset.py); results next to the
                tensor ones under <adv dir>/IMAGE_DATASET/<model>/
  transforms  - held-out detector on every leave-one-out PGD folder with a
                post-processing (resize64 / crop64 / flip / jpeg75) applied to
                the adversarial images; results under <adv dir>/transforms/
  all         - clean, then experiments, then ablation

results.json, lpips_{net}.json and ssim.json sidecars are git-tracked under
$EVAL_OUTPUT_DIR, mirroring the source path so results from different
machines merge without pkl transfer - commit them and `git pull` elsewhere.

Deps: lpips==0.1.4, torchmetrics (SSIM).

Usage:
  source env_set.sh
  python3 src/compute_metrics.py --mode clean --device cuda:0
  python3 src/compute_metrics.py --mode experiments --device cuda:0
  python3 src/compute_metrics.py --mode ablation --device cuda:0
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from generate_noise import build_control_pkl_path, dataset_tag  # noqa: E402

# ─────────────────────────── grid constants ──────────────────────────────────
EPSILONS    = ["8/255", "16/255", "32/255"]
STEP_COUNTS = [10, 20, 30]
STEP_SIZES  = [0.01, 0.03, 0.05]
SEEDS       = [42, 234, 123]
DETECTORS   = [
    "ojha2023", "corvi2023", "cavia2024",
    "chen2024_convnext", "chen2024_clip", "koutlis2024", "wang2020",
]
# Evaluation-only detectors (never in an attack ensemble). Select with
# --detectors for --mode clean / experiments.
NEW_DETECTORS = [
    "vit_lp14_dinov2", "vit_lp14_reg_dinov2",
    "vit_tp16_224_augreg_in21k", "vit_tp16_224_code_augreg_in21k",
    "effort", "aide", "d3", "omniaid", "dda",
]

CSV_COLUMNS = [
    "epsilon", "steps", "alpha", "seed", "held_out",
    "auroc", "f1", "acc",
    "lpips", "lpips_std", "lpips_net", "n_images",
    "ssim", "ssim_std",
    "auroc_clean", "asr",
    "pkl_path",
]


# ─────────────────────────── path helpers ────────────────────────────────────

def eps_str(eps: str) -> str:
    return eps.replace("/", "_")


def build_ablation_pkl_path(output_dir: str, seed: int, dataset_tag_: str, eps: str,
                            steps: int, alpha: float, held_out: str) -> Path:
    ensemble  = [d for d in DETECTORS if d != held_out]
    list_repr = repr(ensemble)
    dirname   = f"adv_raw_{list_repr}_{eps_str(eps)}_{steps}_{alpha}"
    return (
        Path(f"{output_dir}_seed{seed}")
        / "ADV" / dataset_tag_
        / dirname
        / "adv_dataset.pkl"
    )


def natural_dataset_result_path(eval_output_dir: str, pkl: Path, model: str) -> Path:
    """Where evaluate_detector.py --dataset_type dataset actually writes:
    it mirrors path_to_dataset's full absolute path (sans root and
    filename) under --output_dir, regardless of what we'd prefer."""
    path_name = pkl.parts[1:-1]
    return Path(eval_output_dir) / Path(*path_name) / model / "results.json"


def natural_subfolders_result_path(passed_output_dir: str, dataset_path: str,
                                    subset: int, model: str) -> Path:
    """Where evaluate_detector.py --dataset_type subfolders actually writes:
    mirrors dataset_path's full absolute path (sans root), with
    '_{subset}' appended to the last component."""
    mirrored = str(Path(*Path(dataset_path).parts[1:]))
    if subset != -1:
        mirrored += f"_{subset}"
    return Path(passed_output_dir) / mirrored / model / "results.json"


def relocate_result(natural_path: Path, target_path: Path) -> None:
    """evaluate_detector.py always writes to its own natural (machine-path-
    mirrored) location; move the file to our flattened target and prune the
    now-empty scaffolding it created, so re-runs don't pile up empty dirs."""
    if natural_path == target_path:
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(natural_path), str(target_path))
    d = natural_path.parent
    while True:
        try:
            d.rmdir()
        except OSError:
            break
        d = d.parent


def build_results_json_path(eval_output_dir: str, output_dir: str, pkl: Path, model: str) -> Path:
    """Our flattened target location: pkl's directory relative to
    dirname(output_dir), e.g. output_seed42/ADV/ELSA_TEST_200/adv_raw_[...]/
    - drops the machine-specific absolute prefix (the parent of $OUTPUT_DIR),
    keeps the meaningful output[_seed{N}]/... part."""
    adv_root = Path(output_dir).parent
    rel = pkl.parent.relative_to(adv_root)
    return Path(eval_output_dir) / rel / model / "results.json"


def build_lpips_sidecar_path(results_json: Path, net: str) -> Path:
    return results_json.parent / f"lpips_{net}.json"


def build_ssim_sidecar_path(results_json: Path) -> Path:
    return results_json.parent / "ssim.json"


def clean_result_rel(subset: int) -> str:
    """Tag for the clean-baseline results tree: $EVAL_OUTPUT_DIR/clean/<tag>/
    <model>/results.json. Matches the existing committed convention
    (clean/ELSA_1000/<model>/...), which is a curated flat baseline
    independent of any one machine's dataset path - not a mirror of
    evaluate_detector.py's own path derivation."""
    return f"ELSA_{subset}" if subset != -1 else "ELSA_full"


def load_clean_auroc(clean_dir: str, model: str) -> float:
    """AUROC of `model` on the unattacked dataset (ASR baseline), or None if
    the clean results.json for it hasn't been produced/synced yet."""
    path = Path(clean_dir) / model / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["metrics"]["All"]["auc"]


# ─────────────────────────── path parser (ablation) ──────────────────────────

def parse_path_components(results_json: Path):
    """Extract (eps, steps, alpha, seed, held_out) from a results.json path.

    Matches either the PGD grid ("adv_raw_..._{eps}_255_{steps}_{alpha}") or
    a gaussian-noise control ("noise_control_{eps}_255"). Noise has no
    steps/alpha axis - it's parameterized by epsilon alone - so those come
    back as None for noise cells.
    """
    parts = results_json.parts
    held_out = results_json.parent.name

    seed_part = next((p for p in parts if p.startswith("output_seed") or "_seed" in p), None)
    m_seed = re.search(r'_seed(\d+)$', seed_part) if seed_part else None
    if m_seed is None:
        return None
    seed = int(m_seed.group(1))

    adv_dir = next((p for p in parts if p.startswith("adv_raw_")), None)
    if adv_dir is not None:
        m = re.search(r'_(\d+)_255_(\d+)_([\d.]+)$', adv_dir)
        if m is None:
            return None
        eps_num_str, steps_str, alpha_str_v = m.groups()
        return f"{eps_num_str}/255", int(steps_str), float(alpha_str_v), seed, held_out

    noise_dir = next((p for p in parts if p.startswith("noise_control_")), None)
    if noise_dir is not None:
        m = re.match(r'noise_control_(\d+)_255$', noise_dir)
        if m is None:
            return None
        return f"{m.group(1)}/255", None, None, seed, held_out

    return None


# ─────────────────────── AUROC evaluation (subprocess) ──────────────────────

def run_evaluate_detector(pkl: Path, model: str, eval_output_dir: str, output_dir: str,
                          device: str, repo_src: str, batch_size: int = 32,
                          num_workers: int = 4) -> tuple:
    target_path  = build_results_json_path(eval_output_dir, output_dir, pkl, model)
    natural_path = natural_dataset_result_path(eval_output_dir, pkl, model)
    cmd = [
        sys.executable, "raid/evaluate_detector.py",
        "--model",           model,
        "--device",          device,
        "--path_to_dataset", str(pkl),
        "--dataset_type",    "dataset",
        "--output_dir",      eval_output_dir,
        "--batch_size",      str(batch_size),
        "--num_workers",     str(num_workers),
        "--subset",          "-1",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_src}:{repo_src}/raid:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(cmd, cwd=repo_src, env=env, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"  ERROR: evaluate_detector.py failed (rc={proc.returncode})")
        print(proc.stderr[-2000:])
        return None, "error"
    if not natural_path.exists():
        print(f"  ERROR: results.json not written at {natural_path}")
        return None, "error"

    relocate_result(natural_path, target_path)
    with open(target_path) as f:
        data = json.load(f)
    return data["metrics"]["All"], "computed"


# ───────────────────────── LPIPS computation ─────────────────────────────────

def check_pair(clean: torch.Tensor, adv: torch.Tensor, eps_num: float, cls_key: str) -> None:
    """Shapes match and the perturbation respects the L_inf budget (on CPU,
    so full-size classes don't have to fit on the GPU at once)."""
    if clean.shape != adv.shape:
        raise ValueError(f"Shape mismatch '{cls_key}': {clean.shape} vs {adv.shape}")
    max_diff = (adv - clean).abs().max().item()
    if max_diff > eps_num + 1e-4:
        raise RuntimeError(
            f"Sanity FAILED '{cls_key}': max|adv-clean|={max_diff:.6f} > "
            f"eps={eps_num:.6f}+1e-4"
        )


def compute_lpips(adv_data: dict, clean_images: dict, eps_num: float,
                  loss_fn, device: str, batch_size: int = 32) -> tuple:
    per_image = []
    for cls_key in sorted(clean_images.keys()):
        clean = clean_images[cls_key]
        adv   = adv_data[cls_key]["images"]
        check_pair(clean, adv, eps_num, cls_key)
        with torch.no_grad():
            for i in range(0, clean.shape[0], batch_size):
                clean_s = clean[i:i+batch_size].to(device) * 2 - 1
                adv_s   = adv[i:i+batch_size].to(device) * 2 - 1
                vals = loss_fn(clean_s, adv_s)
                v = vals.squeeze().cpu().float()
                per_image.extend(v.tolist() if v.dim() > 0 else [v.item()])
    t = torch.tensor(per_image)
    return t.mean().item(), t.std().item(), len(per_image)


# ───────────────────────── SSIM computation ──────────────────────────────────

def compute_ssim(adv_data: dict, clean_images: dict, eps_num: float,
                 device: str, batch_size: int = 32) -> tuple:
    """Per-image SSIM(clean, adv) on [0,1] RGB, Gaussian 11x11 window, sigma=1.5
    (Wang et al. 2004 defaults), averaged over channels. Higher = more similar."""
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
    per_image = []
    for cls_key in sorted(clean_images.keys()):
        clean = clean_images[cls_key]
        adv   = adv_data[cls_key]["images"]
        check_pair(clean, adv, eps_num, cls_key)
        with torch.no_grad():
            for i in range(0, clean.shape[0], batch_size):
                vals = ssim_fn(adv[i:i+batch_size].to(device).float(),
                               clean[i:i+batch_size].to(device).float(),
                               gaussian_kernel=True, sigma=1.5, kernel_size=11,
                               data_range=1.0, reduction="none")
                per_image.extend(vals.cpu().float().reshape(-1).tolist())
    t = torch.tensor(per_image)
    return t.mean().item(), t.std().item(), len(per_image)


# ───────────────────────── clean image loading (LPIPS/SSIM) ──────────────────

def load_clean_images(clean_dataset: str, subset: int = 200, num_workers: int = 4) -> dict:
    """{folder: [N,3,256,256] in [0,1]}, same order/crop as get_dataloader
    (sorted files, CenterCrop 256). subset=-1 loads every image."""
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader, Subset
    from torchvision.transforms import CenterCrop, ToTensor, Compose

    class PathDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths, self.labels, self.transform = paths, labels, transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            return self.transform(Image.open(self.paths[idx]).convert("RGB")), self.labels[idx]

    transform = Compose([CenterCrop((256, 256)), ToTensor()])
    suffixes  = {".png", ".jpg", ".jpeg"}
    result    = {}
    root      = Path(clean_dataset)
    for folder in sorted(root.iterdir()):
        if folder.name == "modern_generator":
            continue
        paths  = [str(f) for f in sorted(folder.iterdir()) if f.suffix.lower() in suffixes]
        labels = [0 if folder.name == "real" else 1] * len(paths)
        ds     = PathDataset(paths, labels, transform)
        if subset != -1:
            ds = Subset(ds, list(range(subset)))
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers)
        result[folder.name] = torch.cat([b for b, _ in loader], dim=0)
    return result


# ─────────────────────────── CSV rebuild (ablation) ──────────────────────────

def rebuild_csv(eval_output_dir: str, output_csv: str, lpips_net: str = "alex",
                clean_dir: str = "") -> int:
    """Walk eval_output_dir tree, pair results.json + lpips_{net}.json → CSV.

    ASR = (AUROC_clean - AUROC_adv) / AUROC_clean, where AUROC_clean is the
    held-out detector's AUROC on the unattacked dataset (per-detector, not
    per-cell - cached in `clean_auroc_cache`).
    """
    root   = Path(eval_output_dir)
    rows   = []
    sidecar_name = f"lpips_{lpips_net}.json"
    clean_auroc_cache = {}
    n_no_clean = 0
    n_no_ssim = 0

    for rj in sorted(root.rglob("results.json")):
        parsed = parse_path_components(rj)
        if parsed is None:
            continue
        eps, steps, alpha, seed, held_out = parsed

        with open(rj) as f:
            rd = json.load(f)
        m = rd["metrics"]["All"]

        lpips_path = rj.parent / sidecar_name
        if not lpips_path.exists():
            continue
        with open(lpips_path) as f:
            ld = json.load(f)

        ssim_path = rj.parent / "ssim.json"
        sd = {}
        if ssim_path.exists():
            with open(ssim_path) as f:
                sd = json.load(f)
        else:
            n_no_ssim += 1

        if held_out not in clean_auroc_cache:
            clean_auroc_cache[held_out] = load_clean_auroc(clean_dir, held_out)
        auroc_clean = clean_auroc_cache[held_out]
        if auroc_clean is None:
            n_no_clean += 1
            asr = None
        else:
            asr = (auroc_clean - m["auc"]) / auroc_clean

        rows.append({
            "epsilon":     eps,
            "steps":       steps,
            "alpha":       alpha,
            "seed":        seed,
            "held_out":    held_out,
            "auroc":       m["auc"],
            "f1":          m["f1"],
            "acc":         m["acc"],
            "lpips":       ld["lpips"],
            "lpips_std":   ld["lpips_std"],
            "lpips_net":   ld.get("lpips_net", lpips_net),
            "n_images":    ld["n_images"],
            "ssim":        sd.get("ssim"),
            "ssim_std":    sd.get("ssim_std"),
            "auroc_clean": auroc_clean,
            "asr":         asr,
            "pkl_path":    "",
        })

    if n_no_clean:
        print(f"WARNING: {n_no_clean} rows missing clean AUROC "
              f"(no results.json under {clean_dir}/<model>/) - asr left blank.")
    if n_no_ssim:
        print(f"WARNING: {n_no_ssim} rows missing ssim.json - ssim left blank.")

    eps_order = {e: i for i, e in enumerate(EPSILONS)}
    rows.sort(key=lambda r: (
        eps_order.get(str(r["epsilon"]), 99),
        r["steps"] is None,  # noise rows (steps=None) sort after PGD rows
        r["steps"] if r["steps"] is not None else -1,
        r["alpha"] if r["alpha"] is not None else -1.0,
        int(r["seed"]),
        str(r["held_out"]),
    ))

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out_path)
    print(f"CSV written: {out_path} ({len(rows)} rows)")
    return len(rows)


# ─────────────────────── per-cell AUROC + LPIPS (ablation) ───────────────────

def process_cell(pkl: Path, model: str, eps_num: float, args, repo_src: str,
                 state: dict, tag: str) -> bool:
    """Evaluate+LPIPS a single (pkl, model) cell, using/writing the
    results.json + lpips sidecar cache. Shared by the PGD grid loop (one
    ensemble-excluded pkl per held_out model) and the noise grid loop (one
    pkl per epsilon/seed, evaluated by every model the same way)."""
    result_path  = build_results_json_path(args.eval_output_dir, args.output_dir, pkl, model)
    sidecar_path = build_lpips_sidecar_path(result_path, args.lpips_net)
    ssim_path    = build_ssim_sidecar_path(result_path)

    # ── AUROC ─────────────────────────────────────────────────────────
    if result_path.exists() and not args.force:
        with open(result_path) as f:
            auroc_metrics = json.load(f)["metrics"]["All"]
        auroc_status = "auroc-cached"
        state["n_eval_cached"] += 1
    else:
        if args.dry_run:
            print(f"[dry] {tag}: would evaluate + compute LPIPS + SSIM")
            return False
        metrics_all, status = run_evaluate_detector(
            pkl, model, args.eval_output_dir, args.output_dir, args.device, repo_src,
            args.batch_size, args.num_workers)
        if metrics_all is None:
            state["n_eval_error"] += 1
            return False
        auroc_metrics = metrics_all
        auroc_status  = "auroc-computed"
        state["n_eval_computed"] += 1

    need_lpips = args.force or not sidecar_path.exists()
    need_ssim  = args.force or not ssim_path.exists()
    if (need_lpips or need_ssim) and args.dry_run:
        return False

    adv_data = None
    if need_lpips or need_ssim:
        if state["clean_images"] is None:
            print("Loading clean images…")
            state["clean_images"] = load_clean_images(args.test_path, 200)
            print(f"  {sum(v.shape[0] for v in state['clean_images'].values())} images loaded.")
        import pickle
        with open(pkl, "rb") as f:
            adv_data = pickle.load(f)

    # ── LPIPS ─────────────────────────────────────────────────────────
    if not need_lpips:
        with open(sidecar_path) as f:
            sd = json.load(f)
        lpips_mean = sd["lpips"]
        lpips_status = "lpips-cached"
        state["n_lpips_cached"] += 1
    else:
        if state["lpips_loss_fn"] is None:
            import lpips as lpips_lib
            print(f"Building LPIPS model (net={args.lpips_net})…")
            state["lpips_loss_fn"] = lpips_lib.LPIPS(net=args.lpips_net).to(args.device)
            state["lpips_loss_fn"].eval()

        lpips_mean, lpips_std, n_images = compute_lpips(
            adv_data, state["clean_images"], eps_num, state["lpips_loss_fn"], args.device)
        lpips_status = "lpips-computed"
        state["n_lpips_computed"] += 1

        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = sidecar_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({
                "lpips":     lpips_mean,
                "lpips_std": lpips_std,
                "lpips_net": args.lpips_net,
                "n_images":  n_images,
            }, f, indent=2)
        tmp.replace(sidecar_path)

    # ── SSIM ──────────────────────────────────────────────────────────
    if not need_ssim:
        with open(ssim_path) as f:
            ssim_mean = json.load(f)["ssim"]
        ssim_status = "ssim-cached"
        state["n_ssim_cached"] += 1
    else:
        ssim_mean, ssim_std, n_images = compute_ssim(
            adv_data, state["clean_images"], eps_num, args.device)
        ssim_status = "ssim-computed"
        state["n_ssim_computed"] += 1

        ssim_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ssim_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({
                "ssim":        ssim_mean,
                "ssim_std":    ssim_std,
                "kernel":      "gaussian11_sigma1.5",
                "data_range":  1.0,
                "n_images":    n_images,
            }, f, indent=2)
        tmp.replace(ssim_path)

    state["n_rows_written"] += 1
    print(
        f"  OK [{state['n_rows_written']}] {tag} | "
        f"AUROC={auroc_metrics['auc']:.4f} "
        f"LPIPS={lpips_mean:.4f} SSIM={ssim_mean:.4f} "
        f"({auroc_status}, {lpips_status}, {ssim_status})",
        flush=True,
    )
    return True


# ──────────────────────────────── mode: clean ────────────────────────────────

def addgen_clean_rel(args) -> str:
    """Clean-baseline tag for the external-generators dataset, e.g.
    EXTERNAL_GEN_200 - mirrors its ADV/<name>_<subset>/ output dir."""
    return f"{Path(args.addgen_path).name}_{args.addgen_subset}"


def run_clean(args, repo_src: str) -> None:
    """Clean baselines: ELSA (--test_path, --subset) and, if its symlinked
    dataset dir exists (built by run_experiments.sh --mode
    additional_generators), the external generators (--addgen_subset)."""
    run_clean_eval(args, repo_src, args.test_path, args.subset, clean_result_rel(args.subset))
    if args.datasets and not any(d.startswith(Path(args.addgen_path).name) for d in args.datasets):
        return
    if args.addgen_path and Path(args.addgen_path).is_dir():
        run_clean_eval(args, repo_src, args.addgen_path, args.addgen_subset, addgen_clean_rel(args))
    else:
        print(f"No external-generators dataset at {args.addgen_path} - skipping its clean baseline.")


def run_clean_eval(args, repo_src: str, dataset_path: str, subset: int, rel: str) -> None:
    clean_root = str(Path(args.eval_output_dir) / "clean")
    for model in args.detectors:
        target_path = Path(clean_root) / rel / model / "results.json"
        if target_path.exists() and not args.force:
            print(f"SKIP (exists): {target_path}")
            continue
        if args.dry_run:
            print(f"[dry] clean eval: {model}")
            continue

        print(f"==== clean evaluation: {model} ====")
        cmd = [
            sys.executable, "raid/evaluate_detector.py",
            "--model", model,
            "--device", args.device,
            "--path_to_dataset", dataset_path,
            "--dataset_type", "subfolders",
            "--output_dir", clean_root,
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--subset", str(subset),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{repo_src}:{repo_src}/raid:{env.get('PYTHONPATH', '')}"
        proc = subprocess.run(cmd, cwd=repo_src, env=env)
        if proc.returncode != 0:
            print(f"  ERROR: evaluate_detector.py failed for {model} (rc={proc.returncode})")
            continue

        natural_path = natural_subfolders_result_path(clean_root, dataset_path, subset, model)
        if not natural_path.exists():
            print(f"  ERROR: results.json not written at {natural_path}")
            continue
        relocate_result(natural_path, target_path)

    print(f"Clean evaluation complete. Results under {clean_root}/{rel}/")


# ────────────────────────────── mode: experiments ────────────────────────────

def run_experiments_eval(args, repo_src: str) -> None:
    """Evaluate all 7 detectors on every white-box / LOO / full-RAID
    adv_dataset.pkl found under $OUTPUT_DIR/ADV/ (produced by
    run_experiments.sh's "experiments" mode). Discovery-based, so it picks up
    whatever epsilon/step_size/subset combinations were actually generated."""
    pkls = sorted(Path(args.output_dir).glob("ADV/**/adv_*/adv_dataset.pkl"))
    if args.datasets:
        pkls = [p for p in pkls if p.parent.parent.name in args.datasets]
    # --epsilons (default: all) - lets parallel runs split the work by budget
    pkls = [p for p in pkls if any(f"_{eps_str(e)}_" in p.parent.name for e in args.epsilons)]
    if args.attacks:
        pkls = [p for p in pkls if any(p.parent.name.startswith(f"adv_{a}_") for a in args.attacks)]
    if not pkls:
        print(f"No adv_dataset.pkl found under {args.output_dir}/ADV/")
        return

    n_total = n_skip = n_done = 0
    for pkl in pkls:
        detectors = args.detectors
        if args.held_out_only:
            # leave-one-out folders only: evaluate just the detector missing from the ensemble
            src = re.findall(r"'([^']+)'", pkl.parent.name)
            detectors = [d for d in DETECTORS if d not in src] if len(src) == len(DETECTORS) - 1 else []
        for eval_model in detectors:
            n_total += 1
            result_path = build_results_json_path(args.eval_output_dir, args.output_dir, pkl, eval_model)
            if result_path.exists() and not args.force:
                n_skip += 1
                continue
            if args.dry_run:
                print(f"[dry] {eval_model} on {pkl}")
                continue

            print(f"==== evaluating {eval_model} on {pkl} ====")
            _, status = run_evaluate_detector(
                pkl, eval_model, args.eval_output_dir, args.output_dir, args.device, repo_src,
                args.batch_size, args.num_workers)
            if status == "computed":
                n_done += 1

    print(f"Experiments evaluation complete: {n_total} cells, "
          f"{n_done} computed, {n_skip} skipped (exists).")


# ────────────────────────────── mode: subsampling ───────────────────────────

def run_subsampling_eval(args, repo_src: str) -> None:
    """Evaluate all 7 detectors on every adv_dataset.pkl found under
    ${OUTPUT_DIR}_mean/<seed>/ADV/.  Results go to
    ${EVAL_OUTPUT_DIR}/mean/<seed>/ADV/.../<model>/results.json."""
    mean_output_dir = f"{args.output_dir}_mean"
    mean_eval_dir = str(Path(args.eval_output_dir) / "mean")

    mean_root = Path(mean_output_dir)
    if not mean_root.exists():
        print(f"No mean output directory found at {mean_output_dir}")
        return

    seed_dirs = sorted([d for d in mean_root.iterdir() if d.is_dir()])
    if not seed_dirs:
        print(f"No seed subdirectories found under {mean_output_dir}")
        return

    n_total = n_skip = n_done = n_error = 0
    for seed_dir in seed_dirs:
        seed = seed_dir.name
        pkls = sorted(seed_dir.glob("ADV/**/adv_*/adv_dataset.pkl"))
        # --epsilons (default: all) - lets parallel runs split the work by budget
        pkls = [p for p in pkls if any(f"_{eps_str(e)}_" in p.parent.name for e in args.epsilons)]
        if not pkls:
            print(f"  seed={seed}: no pkls found")
            continue

        print(f"\n══════ seed={seed} ({len(pkls)} pkls) ══════")
        for pkl in pkls:
            for eval_model in DETECTORS:
                n_total += 1
                result_path = build_results_json_path(
                    mean_eval_dir, str(seed_dir), pkl, eval_model)
                if result_path.exists() and not args.force:
                    n_skip += 1
                    continue
                if args.dry_run:
                    print(f"[dry] {eval_model} on {pkl}")
                    continue

                print(f"  {eval_model} on {pkl.parent.name} (seed={seed})")
                _, status = run_evaluate_detector(
                    pkl, eval_model, mean_eval_dir, str(seed_dir),
                    args.device, repo_src, args.batch_size, args.num_workers)
                if status == "computed":
                    n_done += 1
                else:
                    n_error += 1

    print(f"\nSubsampling evaluation complete: {n_total} cells, "
          f"{n_done} computed, {n_skip} skipped, {n_error} errors.")
    print(f"Results under {mean_eval_dir}/")


# ───────────────────────────── mode: per_subset ──────────────────────────────

PER_SUBSET_COLUMNS = [
    "group", "dataset", "attack_dir", "eval_model", "subset", "n",
    "acc", "auroc", "auroc_clean", "asr", "results_json",
]


def per_subset_rows(results_json: Path, eval_output_dir: Path, clean_by_model: dict):
    """Split one results.json into per-subset rows. No GPU: uses the stored
    per-sample logits, which evaluate_detector.py writes in dataloader order
    (sorted subset names, equal-size subsets). AUROC per generator is
    real-vs-that-generator; 'real' gets acc only."""
    from sklearn.metrics import roc_auc_score

    with open(results_json) as f:
        d = json.load(f)
    subsets = [k for k in d["metrics"] if k != "All"]
    logits = d["logits"]
    n = len(logits) // len(subsets)
    if n * len(subsets) != len(logits):
        raise ValueError(f"{results_json}: {len(logits)} logits not divisible by {len(subsets)} subsets")
    # float32 softmax, as evaluate_detector.py does: it saturates to ties for
    # confident detectors, and matching it keeps 'All' equal to the stored auc.
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1)[:, 1].tolist()
    scores = {k: probs[i * n:(i + 1) * n] for i, k in enumerate(subsets)}

    rel = results_json.relative_to(eval_output_dir)
    eval_model = rel.parent.name
    if len(rel.parts) > 4:   # <group...>/<dataset>/<attack_dir>/<model>/results.json
        group, dataset, attack_dir = str(Path(*rel.parts[:-4])), rel.parts[-4], rel.parts[-3]
    else:                    # clean/<dataset>/<model>/results.json
        group, dataset, attack_dir = rel.parts[0], rel.parts[-3], "clean"
    clean = clean_by_model.get(eval_model, {})

    rows = []
    for k in subsets + ["All"]:
        row = {"group": group, "dataset": dataset, "attack_dir": attack_dir,
               "eval_model": eval_model, "subset": k,
               "n": n * len(subsets) if k == "All" else n,
               "acc": d["metrics"][k]["acc"], "auroc": "", "auroc_clean": "", "asr": "",
               "results_json": str(rel)}
        if k == "All":
            row["auroc"] = d["metrics"]["All"]["auc"]
        elif k != "real" and "real" in scores:
            y = [0] * len(scores["real"]) + [1] * len(scores[k])
            row["auroc"] = roc_auc_score(y, scores["real"] + scores[k])
        c = clean.get(k, {}).get("auroc")
        if row["auroc"] != "" and c:
            row["auroc_clean"] = c
            row["asr"] = (c - row["auroc"]) / c * 100
        rows.append(row)
    return rows


def run_per_subset(args) -> None:
    """No GPU: per-generator AUROC/acc/ASR for every results.json under
    $EVAL_OUTPUT_DIR, written to one CSV. ASR baseline = the same subset's
    AUROC in the clean results, matching build_results_table.py: ELSA
    generators from --clean_dir, external generators from
    clean/<addgen_clean_rel> (produced by --mode clean; ASR blank if absent)."""
    eval_dir = Path(args.eval_output_dir)
    clean_dirs = [Path(args.clean_dir), eval_dir / "clean" / addgen_clean_rel(args)]

    # {model: {subset: row}}; generator names don't overlap across clean sets.
    # 'All' comes from the first (ELSA) set only - the external set's 'All'
    # would overwrite it; external 'All' rows get their baseline below.
    clean_by_model, clean_all = {}, {}
    for clean_dir in clean_dirs:
        for cj in sorted(clean_dir.glob("*/results.json")):
            rows = {r["subset"]: r for r in per_subset_rows(cj, eval_dir, {})}
            clean_all.setdefault(cj.parent.name, {})[clean_dir.name] = rows.pop("All")
            clean_by_model.setdefault(cj.parent.name, {}).update(rows)

    out_rows, n_err = [], 0
    for rj in sorted(eval_dir.rglob("results.json")):
        try:
            # 'All' baseline = the clean set this dataset was drawn from
            clean_tag = next((c.name for c in clean_dirs[1:]
                              if rj.relative_to(eval_dir).parts[-4:-3] == (c.name,)),
                             clean_dirs[0].name)
            baseline = {m: {**subs, "All": clean_all.get(m, {}).get(clean_tag, {})}
                        for m, subs in clean_by_model.items()}
            out_rows.extend(per_subset_rows(rj, eval_dir, baseline))
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"[err] {rj}: {e}")

    out = Path(args.output or Path(eval_dir).parent / "per_subset_metrics.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_SUBSET_COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Per-subset metrics: {len(out_rows)} rows, {n_err} errors -> {out}")


# ──────────────────────────────── mode: ablation ─────────────────────────────

def run_ablation(args) -> None:
    output_csv = args.output or str(Path(args.results_dir) / "ablation_metrics.csv")

    if args.rebuild_csv_only:
        n = rebuild_csv(args.eval_output_dir, output_csv, args.lpips_net, args.clean_dir)
        print(f"rebuild_csv_only: {n} rows from {args.eval_output_dir}")
        return

    repo_src = str(Path(__file__).parent)
    tag = dataset_tag(args.test_path, 200)
    state = {
        "clean_images": None, "lpips_loss_fn": None,
        "n_eval_computed": 0, "n_eval_cached": 0, "n_eval_error": 0,
        "n_lpips_computed": 0, "n_lpips_cached": 0,
        "n_ssim_computed": 0, "n_ssim_cached": 0, "n_rows_written": 0,
    }

    n_total = n_skip_no_pkl = 0

    if not args.noise_only:
        for eps in args.epsilons:
            eps_num = float(eval(eps))
            for steps in args.steps:
                for alpha in args.alphas:
                    for seed in args.seeds:
                        for held_out in DETECTORS:
                            n_total += 1
                            pkl = build_ablation_pkl_path(
                                args.output_dir, seed, tag, eps, steps, alpha, held_out)
                            cell_tag = (f"eps={eps} steps={steps} alpha={alpha} "
                                   f"seed={seed} held_out={held_out}")

                            if not pkl.exists():
                                print(f"SKIP (no pkl): {cell_tag}")
                                n_skip_no_pkl += 1
                                continue

                            process_cell(pkl, held_out, eps_num, args, repo_src, state, cell_tag)

    n_noise_total = n_noise_skip_no_pkl = 0
    if not args.no_noise:
        for eps in args.epsilons:
            eps_num = float(eval(eps))
            for seed in args.seeds:
                pkl = build_control_pkl_path(args.output_dir, seed, tag, eps)
                noise_tag = f"[noise] eps={eps} seed={seed}"

                if not pkl.exists():
                    print(f"SKIP (no pkl): {noise_tag}")
                    n_noise_skip_no_pkl += len(DETECTORS)
                    n_noise_total += len(DETECTORS)
                    continue

                for model in DETECTORS:
                    n_noise_total += 1
                    process_cell(pkl, model, eps_num, args, repo_src, state,
                                f"{noise_tag} model={model}")

    if not args.dry_run:
        rebuild_csv(args.eval_output_dir, output_csv, args.lpips_net, args.clean_dir)
    else:
        print(f"\n[dry_run] {n_total + n_noise_total} cells | "
              f"skip: {n_skip_no_pkl + n_noise_skip_no_pkl} | "
              f"processable: {n_total + n_noise_total - n_skip_no_pkl - n_noise_skip_no_pkl}")

    print("\n" + "=" * 60)
    print(f"  PGD grid cells:    {n_total}  (skipped no-pkl: {n_skip_no_pkl})")
    if not args.no_noise:
        print(f"  Noise grid cells:  {n_noise_total}  (skipped no-pkl: {n_noise_skip_no_pkl})")
    print(f"  Rows written:      {state['n_rows_written']}")
    print(f"  AUROC computed:    {state['n_eval_computed']}  "
          f"cached: {state['n_eval_cached']}  errors: {state['n_eval_error']}")
    print(f"  LPIPS computed:    {state['n_lpips_computed']}  cached: {state['n_lpips_cached']}")
    print(f"  SSIM computed:     {state['n_ssim_computed']}  cached: {state['n_ssim_cached']}")


# ─────────────────────────────── mode: transforms ────────────────────────────

TRANSFORMS = ["resize64", "crop64", "flip", "jpeg75"]


def run_transforms(args, repo_src: str) -> None:
    """Held-out detector on every leave-one-out PGD folder (--datasets, default
    ELSA_TEST_1000), with each --transforms post-processing applied to the
    adversarial images before the detector. Results land next to the
    untransformed ones as <adv dir>/transforms/<transform>/<model>/results.json;
    ASR is computed against the untransformed clean AUROC (build_results_table)."""
    datasets = args.datasets or ["ELSA_TEST_1000"]
    pkls = sorted(p for d in datasets
                  for p in (Path(args.output_dir) / "ADV" / d).glob("adv_raw_*/adv_dataset.pkl"))
    pkls = [p for p in pkls if any(f"_{eps_str(e)}_" in p.parent.name for e in args.epsilons)]
    n_done = n_skip = n_err = 0
    for pkl in pkls:
        src = re.findall(r"'([^']+)'", pkl.parent.name)
        if len(src) != len(DETECTORS) - 1:
            continue  # leave-one-out folders only
        held_out = next(d for d in DETECTORS if d not in src)
        for transform in args.transforms:
            target = (build_results_json_path(args.eval_output_dir, args.output_dir, pkl, held_out)
                      .parent.parent / "transforms" / transform / held_out / "results.json")
            if target.exists() and not args.force:
                n_skip += 1
                continue
            if args.dry_run:
                print(f"[dry] {held_out} / {transform} on {pkl.parent.name}")
                continue
            print(f"==== {held_out} / {transform} on {pkl.parent.name} ====", flush=True)
            cmd = [
                sys.executable, "raid/evaluate_detector.py",
                "--model", held_out,
                "--device", args.device,
                "--path_to_dataset", str(pkl),
                "--dataset_type", "dataset",
                "--output_dir", args.eval_output_dir,
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--subset", "-1",
                "--transform", transform,
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{repo_src}:{repo_src}/raid:{env.get('PYTHONPATH', '')}"
            proc = subprocess.run(cmd, cwd=repo_src, env=env)
            natural = natural_dataset_result_path(args.eval_output_dir, pkl, f"{held_out}__{transform}")
            if proc.returncode != 0 or not natural.exists():
                print(f"  ERROR: {held_out} / {transform} on {pkl} (rc={proc.returncode})")
                n_err += 1
                continue
            relocate_result(natural, target)
            n_done += 1
    print(f"Transforms evaluation: {n_done} computed, {n_skip} skipped (exists), {n_err} errors.")


# ─────────────────────────────── mode: raid_img ──────────────────────────────

def run_raid_img(args, repo_src: str) -> None:
    """Evaluate --detectors on the PNG-quantized RAID release: every
    $OUTPUT_DIR/ADV/<basename $TEST_PATH>/adv_*/IMAGE_DATASET/ (one subfolder
    per class, as the clean test set). Results land in
    $EVAL_OUTPUT_DIR/output/ADV/<dataset>/<adv dir>/IMAGE_DATASET/<model>/."""
    dataset = Path(args.test_path).name
    img_dirs = sorted((Path(args.output_dir) / "ADV" / dataset).glob("adv_*/IMAGE_DATASET"))
    img_dirs = [d for d in img_dirs if any(f"_{eps_str(e)}_" in d.parent.name for e in args.epsilons)]
    if args.attacks:
        img_dirs = [d for d in img_dirs
                    if any(d.parent.name.startswith(f"adv_{a}_") for a in args.attacks)]
    if not img_dirs:
        print(f"No IMAGE_DATASET dir under {args.output_dir}/ADV/{dataset}/adv_*/")
        return

    n_done = n_skip = n_err = 0
    for img_dir in img_dirs:
        # same mirroring as the tensor results: <adv dir>/IMAGE_DATASET/<model>/
        img_root = build_results_json_path(args.eval_output_dir, args.output_dir,
                                           img_dir / "_", "_").parent.parent
        for model in args.detectors:
            target_path = img_root / model / "results.json"
            if target_path.exists() and not args.force:
                n_skip += 1
                continue
            if args.dry_run:
                print(f"[dry] {model} on {img_dir}")
                continue
            print(f"==== {model} on {img_dir.parent.name}/IMAGE_DATASET ====", flush=True)
            cmd = [
                sys.executable, "raid/evaluate_detector.py",
                "--model", model,
                "--device", args.device,
                "--path_to_dataset", str(img_dir),
                "--dataset_type", "subfolders",
                "--output_dir", args.eval_output_dir,
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--subset", "-1",
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{repo_src}:{repo_src}/raid:{env.get('PYTHONPATH', '')}"
            proc = subprocess.run(cmd, cwd=repo_src, env=env)
            natural_path = natural_subfolders_result_path(args.eval_output_dir, str(img_dir), -1, model)
            if proc.returncode != 0 or not natural_path.exists():
                print(f"  ERROR: {model} on {img_dir} (rc={proc.returncode})")
                n_err += 1
                continue
            relocate_result(natural_path, target_path)
            n_done += 1
    print(f"RAID img evaluation: {n_done} computed, {n_skip} skipped (exists), {n_err} errors.")


# ─────────────────────────── mode: raid_perceptual ───────────────────────────

def run_raid_perceptual(args) -> None:
    """LPIPS + SSIM of the full RAID dataset (every adv_*/adv_dataset.pkl
    under $OUTPUT_DIR/ADV/<basename $TEST_PATH>/) against the full clean test
    set. Detector-independent, so one lpips_{net}.json + ssim.json per pkl,
    written next to that pkl's per-detector results dirs under
    $EVAL_OUTPUT_DIR. Each pkl is ~19 GB and the clean set another ~19 GB,
    all held in RAM."""
    import gc
    import pickle

    dataset = Path(args.test_path).name
    pkls = sorted((Path(args.output_dir) / "ADV" / dataset).glob("adv_*/adv_dataset.pkl"))
    pkls = [p for p in pkls if any(f"_{eps_str(e)}_" in p.parent.name for e in args.epsilons)]
    if args.attacks:
        pkls = [p for p in pkls if any(p.parent.name.startswith(f"adv_{a}_") for a in args.attacks)]
    if not pkls:
        print(f"No adv_dataset.pkl found under {args.output_dir}/ADV/{dataset}/")
        return

    clean_images = loss_fn = None
    for pkl in pkls:
        out_dir    = build_results_json_path(
            args.eval_output_dir, args.output_dir, pkl, "_").parent.parent
        lpips_path = out_dir / f"lpips_{args.lpips_net}.json"
        ssim_path  = out_dir / "ssim.json"
        if lpips_path.exists() and ssim_path.exists() and not args.force:
            print(f"SKIP (cached): {pkl.parent.name}")
            continue
        if args.dry_run:
            print(f"[dry] would compute LPIPS + SSIM: {pkl.parent.name}")
            continue

        m = re.search(r"_(\d+)_255_", pkl.parent.name)
        eps_num = int(m.group(1)) / 255

        if clean_images is None:
            print(f"Loading full clean set {args.test_path}…")
            clean_images = load_clean_images(args.test_path, -1, args.num_workers)
            print(f"  {sum(v.shape[0] for v in clean_images.values())} images loaded.")
        if loss_fn is None:
            import lpips as lpips_lib
            loss_fn = lpips_lib.LPIPS(net=args.lpips_net).to(args.device).eval()

        print(f"==== {pkl.parent.name} ====", flush=True)
        with open(pkl, "rb") as f:
            adv_data = pickle.load(f)

        lp_mean, lp_std, n = compute_lpips(adv_data, clean_images, eps_num, loss_fn,
                                           args.device, args.batch_size)
        ss_mean, ss_std, _ = compute_ssim(adv_data, clean_images, eps_num,
                                          args.device, args.batch_size)
        del adv_data
        gc.collect()

        out_dir.mkdir(parents=True, exist_ok=True)
        for path, payload in [
            (lpips_path, {"lpips": lp_mean, "lpips_std": lp_std,
                          "lpips_net": args.lpips_net, "n_images": n}),
            (ssim_path,  {"ssim": ss_mean, "ssim_std": ss_std,
                          "kernel": "gaussian11_sigma1.5", "data_range": 1.0,
                          "n_images": n}),
        ]:
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(path)
        print(f"  LPIPS={lp_mean:.4f}±{lp_std:.4f}  SSIM={ss_mean:.4f}±{ss_std:.4f}  "
              f"(n={n})", flush=True)


# ──────────────────────────────── main ───────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate detectors and compute metrics (clean / experiments / ablation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["clean", "experiments", "subsampling", "ablation", "per_subset",
                            "raid_perceptual", "raid_img", "transforms", "all"], required=True)

    p.add_argument("--test_path", default=os.environ.get("TEST_PATH"),
                   help="Clean dataset path (default: $TEST_PATH)")
    p.add_argument("--output_dir", default=os.environ.get("OUTPUT_DIR"),
                   help="Adversarial dataset root (default: $OUTPUT_DIR)")
    p.add_argument("--eval_output_dir", default=os.environ.get("EVAL_OUTPUT_DIR"),
                   help="Git-tracked dir for results.json + lpips sidecars (default: $EVAL_OUTPUT_DIR)")
    p.add_argument("--results_dir", default=os.environ.get("RESULTS_DIR"),
                   help="Dir for the ablation CSV (default: $RESULTS_DIR)")
    p.add_argument("--clean_dir", default=None,
                   help="Dir with clean-baseline results.json/<model>/, used for ablation ASR "
                        "(default: $EVAL_OUTPUT_DIR/clean/<clean_result_rel>)")

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--subset", type=int, default=1000, help="Subset size for --mode clean")
    p.add_argument("--detectors", nargs="+", default=DETECTORS,
                   help="(--mode clean|experiments) detectors to evaluate; "
                        f"evaluation-only extras: {' '.join(NEW_DETECTORS)}")
    p.add_argument("--attacks", nargs="+", default=None,
                   help="(--mode experiments|raid_perceptual|raid_img) only these adv_<attack>_ dirs: "
                        "raw (PGD), apgd, cwa. Default: all")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="(--mode clean|experiments) only these ADV/<dataset> dirs, e.g. "
                        "ELSA_TEST (full RAID) ELSA_TEST_1000 EXTERNAL_GEN_200. For clean, "
                        "the external baseline runs only if EXTERNAL_GEN_* is listed. Default: all")
    p.add_argument("--addgen_path", default=os.environ.get("ADDGEN_PATH"),
                   help="External-generators dataset dir from run_experiments.sh --mode "
                        "additional_generators (default: $ADDGEN_PATH, else "
                        "dirname($TEST_PATH)/EXTERNAL_GEN)")
    p.add_argument("--addgen_subset", type=int, default=200,
                   help="Per-folder subset of the external generators (matches --sub_subset)")

    p.add_argument("--epsilons", nargs="+", default=EPSILONS,
                   help="(--mode ablation|experiments|subsampling) budgets to process, e.g. 16/255")
    p.add_argument("--steps",    type=int,   nargs="+", default=STEP_COUNTS)
    p.add_argument("--alphas",   type=float, nargs="+", default=STEP_SIZES)
    p.add_argument("--seeds",    type=int,   nargs="+", default=SEEDS)

    p.add_argument("--transforms", nargs="+", default=TRANSFORMS, choices=TRANSFORMS,
                   help="(--mode transforms) post-processing to apply to the adversarial images")
    p.add_argument("--held_out_only", action="store_true",
                   help="(--mode experiments) on leave-one-out folders, evaluate only the held-out detector; skip other folders")
    p.add_argument("--force",            action="store_true")
    p.add_argument("--dry_run",          action="store_true")
    p.add_argument("--rebuild_csv_only", action="store_true",
                   help="(--mode ablation) Walk eval_output_dir and rebuild CSV. No GPU, no pkl access.")
    p.add_argument("--no_noise", action="store_true",
                   help="(--mode ablation) Skip the gaussian-noise control grid.")
    p.add_argument("--noise_only", action="store_true",
                   help="(--mode ablation) Only evaluate the noise-control grid.")
    p.add_argument("--output",    default=None, help="(--mode ablation|per_subset) CSV output path override")
    p.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    return p.parse_args()


def main():
    args = parse_args()
    if not args.test_path:
        sys.exit("Set TEST_PATH (see env_set.sh) or pass --test_path")
    if not args.output_dir:
        sys.exit("Set OUTPUT_DIR (see env_set.sh) or pass --output_dir")
    if not args.eval_output_dir:
        sys.exit("Set EVAL_OUTPUT_DIR (see env_set.sh) or pass --eval_output_dir")
    if args.mode in ("ablation", "all") and not args.results_dir:
        sys.exit("Set RESULTS_DIR (see env_set.sh) or pass --results_dir")

    if args.addgen_path is None:
        args.addgen_path = str(Path(args.test_path).parent / "EXTERNAL_GEN")
    if args.clean_dir is None:
        args.clean_dir = str(Path(args.eval_output_dir) / "clean" / clean_result_rel(args.subset))

    repo_src = str(Path(__file__).parent)

    if args.mode in ("clean", "all"):
        run_clean(args, repo_src)
    if args.mode in ("experiments", "all"):
        run_experiments_eval(args, repo_src)
    if args.mode == "subsampling":
        run_subsampling_eval(args, repo_src)
    if args.mode in ("ablation", "all"):
        run_ablation(args)
    if args.mode == "per_subset":
        run_per_subset(args)
    if args.mode == "raid_perceptual":
        run_raid_perceptual(args)
    if args.mode == "raid_img":
        run_raid_img(args, repo_src)
    if args.mode == "transforms":
        run_transforms(args, repo_src)


if __name__ == "__main__":
    main()
