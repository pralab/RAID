#!/usr/bin/env python3
"""Generate noise baselines used by the PGD ablation analysis.

Two kinds of noise, both L-inf bounded by the same epsilon budgets used for
the PGD ablation grid:

  control  - clamped gaussian noise, generated per-seed (matching the PGD
             grid's 3-seed aggregation), written into the same seeded tree
             run_experiments.sh's ablation mode writes into:
               <OUTPUT_DIR>_seed{seed}/ADV/<dataset>_<subset>/
                   noise_control_{eps}/adv_dataset.pkl

  baseline - gaussian and uniform noise, single seed, written into the
             unseeded tree run_experiments.sh's experiments mode writes into:
               <OUTPUT_DIR>/ADV/<dataset>_<subset>/
                   noise_{gaussian,uniform}_{eps}/adv_dataset.pkl

Both reuse raid/data's get_dataloader (same subfolders pipeline as attacks:
sorted folders, CenterCrop(256), ToTensor, Subset(range(subset))).

Usage:
  source env_set.sh
  python3 src/generate_noise.py --type control
  python3 src/generate_noise.py --type baseline
  python3 src/generate_noise.py --type all
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "raid"))

from data import get_dataloader  # noqa: E402
from generate_noise_baseline import add_noise  # noqa: E402

EPSILONS = ["8/255", "16/255", "32/255"]
SEEDS = [42, 234, 123]
SUBSET = 200


def load_clean_dataset(dataset_path: str, subset: int) -> dict:
    """Loads the clean dataset into per-class {images, labels} tensors."""
    data_loaders, _ = get_dataloader(
        path=dataset_path,
        dataset_type="subfolders",
        batch_size=64,
        device="cpu",
        subset=subset,
    )
    result = {}
    for cls_key, loader in data_loaders.items():
        images, labels = zip(*[(b, l) for b, l in loader])
        result[cls_key] = {
            "images": torch.cat(images, dim=0),
            "labels": torch.cat(labels, dim=0),
        }
    return result


def dataset_tag(dataset_path: str, subset: int) -> str:
    name = dataset_path.rstrip("/").split("/")[-1]
    return f"{name}_{subset}" if subset != -1 else name


def build_control_pkl_path(output_dir: str, seed: int, dataset_tag_: str, eps: str) -> Path:
    eps_tag = eps.replace("/", "_")
    return (
        Path(f"{output_dir}_seed{seed}")
        / "ADV" / dataset_tag_
        / f"noise_control_{eps_tag}"
        / "adv_dataset.pkl"
    )


def build_baseline_pkl_path(output_dir: str, dataset_tag_: str, noise_type: str, eps: str) -> Path:
    eps_tag = eps.replace("/", "_")
    return (
        Path(output_dir)
        / "ADV" / dataset_tag_
        / f"noise_{noise_type}_{eps_tag}"
        / "adv_dataset.pkl"
    )


def save_pkl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.replace(path)
    print(f"  Saved: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def run_control(test_path: str, output_dir: str, epsilons: list, seeds: list,
                 subset: int, force: bool) -> None:
    tag = dataset_tag(test_path, subset)
    print(f"Loading clean images from {test_path} (subset={subset})...")
    clean_data = load_clean_dataset(test_path, subset)
    n_images = sum(v["images"].shape[0] for v in clean_data.values())
    print(f"  {n_images} images across {len(clean_data)} classes")

    for eps_str in epsilons:
        eps_num = float(eval(eps_str))
        for seed in seeds:
            out_path = build_control_pkl_path(output_dir, seed, tag, eps_str)
            if out_path.exists() and not force:
                print(f"SKIP (exists): {out_path}")
                continue

            print(f"Generating noise control eps={eps_str} seed={seed} (eps={eps_num:.6f})...")
            gen = torch.Generator().manual_seed(seed)
            noisy_data = {}
            for cls_key, entry in clean_data.items():
                images = entry["images"].clone()
                noise = torch.randn(images.shape, generator=gen) * eps_num
                noise = noise.clamp(-eps_num, eps_num)
                images = (images + noise).clamp(0.0, 1.0)
                noisy_data[cls_key] = {"images": images, "labels": entry["labels"].clone()}

                diff = (images - entry["images"]).abs().max().item()
                assert diff <= eps_num + 1e-6, f"Noise exceeds eps for {cls_key}: {diff}"

            save_pkl(out_path, noisy_data)


def run_baseline(test_path: str, output_dir: str, epsilons: list, subset: int,
                  seed: int, force: bool) -> None:
    tag = dataset_tag(test_path, subset)

    for eps_str in epsilons:
        eps_num = float(eval(eps_str))
        for noise_type in ("gaussian", "uniform"):
            out_path = build_baseline_pkl_path(output_dir, tag, noise_type, eps_str)
            if out_path.exists() and not force:
                print(f"SKIP (exists): {out_path}")
                continue

            print(f"Generating {noise_type} baseline eps={eps_str}...")
            data_loaders, _ = get_dataloader(
                path=test_path,
                dataset_type="subfolders",
                batch_size=64,
                device="cpu",
                subset=subset,
            )
            adv_dataset = {}
            for cls_key, loader in data_loaders.items():
                all_images, all_labels = [], []
                for images, labels in loader:
                    all_images.append(add_noise(images, eps_num, noise_type, seed=seed))
                    all_labels.append(labels)
                adv_dataset[cls_key] = {
                    "images": torch.cat(all_images, dim=0),
                    "labels": torch.cat(all_labels, dim=0),
                }
                print(f"  {cls_key}: {adv_dataset[cls_key]['images'].shape[0]} images")

            save_pkl(out_path, adv_dataset)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate noise baselines for the PGD ablation analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--type", choices=["control", "baseline", "all"], default="all")
    p.add_argument("--test_path", default=os.environ.get("TEST_PATH"),
                   help="Clean dataset path (default: $TEST_PATH)")
    p.add_argument("--output_dir", default=os.environ.get("OUTPUT_DIR"),
                   help="Adversarial dataset output root (default: $OUTPUT_DIR)")
    p.add_argument("--epsilons", nargs="+", default=EPSILONS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                   help="Seeds for --type control (one pkl per seed)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for --type baseline (single pkl per epsilon)")
    p.add_argument("--subset", type=int, default=SUBSET)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.test_path:
        sys.exit("Set TEST_PATH (see env_set.sh) or pass --test_path")
    if not args.output_dir:
        sys.exit("Set OUTPUT_DIR (see env_set.sh) or pass --output_dir")

    if args.type in ("control", "all"):
        run_control(args.test_path, args.output_dir, args.epsilons, args.seeds,
                     args.subset, args.force)
    if args.type in ("baseline", "all"):
        run_baseline(args.test_path, args.output_dir, args.epsilons, args.subset,
                      args.seed, args.force)

    print("Done.")


if __name__ == "__main__":
    main()
