#!/usr/bin/env python3
"""Plot AUROC / ASR vs LPIPS (and/or SSIM) for the PGD ablation grid.

Walks RESULTS_DIR ($EVAL_OUTPUT_DIR, results/evaluate_detector/ at the repo
root) for results.json +
lpips_{net}.json pairs, parses (epsilon, steps, alpha, seed, held_out) from the
path, and plots one point per (epsilon, steps, alpha) triple averaged over all
available (seed, held_out) rows.

Same epsilon → same colour + dashed connector sorted by the x metric.
--x_metric picks the x axis: lpips (default), ssim (from ssim.json sidecars,
written by compute_metrics.py), or config (ASR vs the (steps, alpha) grid,
one line per epsilon + a gaussian-noise line per epsilon).
No GPU, no model loading.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

import shutil as _shutil
if _shutil.which("latex"):
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsfonts}",
    })
else:
    # No LaTeX binary — Computer Modern mathtext gives an equivalent look.
    matplotlib.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
    })

# Base font sizing (bumped up across the board — axis labels, ticks, etc.);
# per-element sizes below (annotations, legend) are set explicitly and scaled
# up to match.
matplotlib.rcParams.update({
    "font.size":        14,
    "axes.labelsize":   16,
    "axes.titlesize":   16,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "legend.fontsize":  12,
})

# ────────────────────────── configurable constants ────────────────────────────

# EVAL_OUTPUT_DIR / RESULTS_DIR match compute_metrics.py's env vars (see
# env_set.sh): the former holds results.json + lpips sidecars to walk, the
# latter is where this script's own figures are written.
EVAL_OUTPUT_DIR = os.environ.get("EVAL_OUTPUT_DIR", "../results/evaluate_detector")
OUTPUT_DIR      = os.environ.get("RESULTS_DIR", "../results/ablation")
CLEAN_DIR       = str(Path(EVAL_OUTPUT_DIR) / "clean" / "ELSA_1000")

EPSILON_COLORS = {
    "8/255":  "#0072B2",
    "16/255": "#f2870d",
    "32/255": "#CC79A7",
}
DEFAULT_COLOR  = "#888888"
EPSILON_ORDER  = ["8/255", "16/255", "32/255"]

ALL_EPSILONS    = ["8/255", "16/255", "32/255"]
ALL_STEP_COUNTS = [10, 20, 30]
ALL_STEP_SIZES  = [0.01, 0.03, 0.05]
ALL_SEEDS       = [42, 234, 123]
ALL_DETECTORS   = [
    "ojha2023", "corvi2023", "cavia2024",
    "chen2024_convnext", "chen2024_clip", "koutlis2024", "wang2020",
]
EXPECTED_ROWS = len(ALL_SEEDS) * len(ALL_DETECTORS)   # 21


# ───────────────────────── tree walking + parsing ─────────────────────────────

def parse_path_components(results_json: Path):
    """Return (eps, steps, alpha, seed, held_out) or None if path doesn't
    match either the PGD grid or a gaussian-noise control. Noise has no
    steps/alpha axis — it's parameterized by epsilon alone — so those come
    back as None for noise cells."""
    parts    = results_json.parts
    held_out = results_json.parent.name

    seed_part = next((p for p in parts if p.startswith("output_seed")), None)
    if seed_part is None:
        return None
    seed = int(seed_part[len("output_seed"):])

    adv_dir = next((p for p in parts if p.startswith("adv_raw_")), None)
    if adv_dir is not None:
        m = re.search(r'_(\d+)_255_(\d+)_([\d.]+)$', adv_dir)
        if m is None:
            return None
        eps_num, steps_str, alpha_str = m.groups()
        return f"{eps_num}/255", int(steps_str), float(alpha_str), seed, held_out

    noise_dir = next((p for p in parts if p.startswith("noise_control_")), None)
    if noise_dir is not None:
        m = re.match(r'noise_control_(\d+)_255$', noise_dir)
        if m is None:
            return None
        return f"{m.group(1)}/255", None, None, seed, held_out

    return None


def load_clean_auroc(clean_dir: str, model: str) -> float:
    """AUROC of `model` on the unattacked dataset (ASR baseline), or None if
    the clean results.json for it hasn't been produced/synced yet."""
    path = Path(clean_dir) / model / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["metrics"]["All"]["auc"]


def load_data(results_dir: str, lpips_net: str, clean_dir: str,
             epsilons=None, seeds=None):
    root   = Path(results_dir)
    sidecar_name = f"lpips_{lpips_net}.json"
    rows   = []
    skipped = 0
    clean_auroc_cache = {}
    n_no_clean = 0

    for rj in sorted(root.rglob("results.json")):
        parsed = parse_path_components(rj)
        if parsed is None:
            skipped += 1
            continue
        eps, steps, alpha, seed, held_out = parsed

        if epsilons and eps not in epsilons:
            continue
        if seeds and seed not in seeds:
            continue

        sidecar = rj.parent / sidecar_name
        if not sidecar.exists():
            continue

        with open(rj) as f:
            rd = json.load(f)
        with open(sidecar) as f:
            ld = json.load(f)

        ssim = None
        ssim_path = rj.parent / "ssim.json"
        if ssim_path.exists():
            with open(ssim_path) as f:
                ssim = json.load(f)["ssim"]

        auroc = rd["metrics"]["All"]["auc"]

        if held_out not in clean_auroc_cache:
            clean_auroc_cache[held_out] = load_clean_auroc(clean_dir, held_out)
        auroc_clean = clean_auroc_cache[held_out]
        if auroc_clean is None:
            n_no_clean += 1
            asr = None
        else:
            asr = (auroc_clean - auroc) / auroc_clean

        rows.append({
            "epsilon": eps, "steps": steps, "alpha": alpha,
            "seed": seed, "held_out": held_out,
            "auroc": auroc,
            "lpips": ld["lpips"],
            "ssim": ssim,
            "asr": asr,
        })

    if skipped:
        print(f"Skipped {skipped} results.json files (path didn't match pattern).")
    if n_no_clean:
        print(f"WARNING: {n_no_clean} rows missing clean AUROC "
              f"(no results.json under {clean_dir}/<model>/) — asr left as None.")
    return rows


def aggregate(rows: list, min_runs: int = 0):
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        key = (r["epsilon"], r["steps"], r["alpha"])
        buckets[key].append(r)

    agg = []
    warned = False
    # steps=None (noise cells) sorts after PGD cells within the same epsilon —
    # comparing None to int directly would raise TypeError.
    sort_key = lambda kv: (kv[0][0], kv[0][1] is None,
                           kv[0][1] if kv[0][1] is not None else -1,
                           kv[0][2] if kv[0][2] is not None else -1.0)
    for (eps, steps, alpha), bucket in sorted(buckets.items(), key=sort_key):
        n = len(bucket)
        if n < EXPECTED_ROWS:
            if not warned:
                print(f"WARNING: some triples have < {EXPECTED_ROWS} rows "
                      f"(7 detectors × {len(ALL_SEEDS)} seeds).")
                warned = True
            alpha_str = f"{alpha:.2f}" if alpha is not None else "  - "
            print(f"  eps={eps} steps={steps} alpha={alpha_str} → {n}/{EXPECTED_ROWS}")
        if min_runs > 0 and n < min_runs:
            continue
        lpips_vals = [r["lpips"] for r in bucket]
        auroc_vals = [r["auroc"] for r in bucket]
        ssim_vals  = [r["ssim"] for r in bucket if r["ssim"] is not None]
        asr_vals   = [r["asr"] for r in bucket if r["asr"] is not None]
        agg.append({
            "epsilon":    eps,
            "steps":      steps,
            "alpha":      alpha,
            "lpips_mean": float(np.mean(lpips_vals)),
            "lpips_std":  float(np.std(lpips_vals)),
            "ssim_mean":  float(np.mean(ssim_vals)) if ssim_vals else None,
            "ssim_std":   float(np.std(ssim_vals)) if ssim_vals else None,
            "auroc_mean": float(np.mean(auroc_vals)),
            "auroc_std":  float(np.std(auroc_vals)),
            "asr_mean":   float(np.mean(asr_vals)) if asr_vals else None,
            "asr_std":    float(np.std(asr_vals)) if asr_vals else None,
            "n_rows":     n,
        })
    return agg


# ─────────────────────────────── plotting ────────────────────────────────────

def draw(ax, agg: list, args: argparse.Namespace, *,
         y_key: str, y_label: str, y_lim: tuple,
         x_key: str = "lpips", x_label: str = "LPIPS", x_lim: tuple = (0.05, 0.5),
         invert_x: bool = False, legend: bool = True,
         scale: float = 1.0, text_scale: float = 1.0) -> bool:
    """Draw y_key ("auroc" or "asr") vs x_key ("lpips" or "ssim") on ax.
    invert_x flips the axis so that for SSIM (higher = more similar)
    perturbation strength still grows left→right, as with LPIPS.
    scale multiplies marker sizes and line widths; text_scale the point-label
    font size.
    Returns False (nothing drawn) if no row has both metrics."""
    y_mean_key, y_std_key = f"{y_key}_mean", f"{y_key}_std"
    x_mean_key, x_std_key = f"{x_key}_mean", f"{x_key}_std"
    agg = [r for r in agg
           if r.get(y_mean_key) is not None and r.get(x_mean_key) is not None]
    if not agg:
        print(f"No rows with {y_mean_key} and {x_mean_key} — "
              f"skipping {y_label} vs {x_label} plot.")
        return False

    # Print table
    print(f"\n{'epsilon':<8} {'steps':>5} {'alpha':>6}  "
          f"{x_label:>8} {'±':>7}  {y_label:>7} {'±':>7}  {'n':>4}")
    print("-" * 62)
    for eps in EPSILON_ORDER:
        sub = [r for r in agg if r["epsilon"] == eps]
        sub.sort(key=lambda r: r[x_mean_key])
        for row in sub:
            steps_str = row["steps"] if row["steps"] is not None else "noise"
            alpha_str = f"{row['alpha']:.2f}" if row["alpha"] is not None else "  - "
            print(f"{row['epsilon']:<8} {steps_str!s:>5} {alpha_str:>6}  "
                  f"{row[x_mean_key]:>8.4f} {row[x_std_key]:>7.4f}  "
                  f"{row[y_mean_key]:>7.4f} {row[y_std_key]:>7.4f}  "
                  f"{row['n_rows']:>4}")

    epsilons_in_data = {r["epsilon"] for r in agg}
    for eps in ALL_EPSILONS:
        if eps not in epsilons_in_data:
            print(f"WARNING: eps={eps} absent — not generated yet on this machine.")

    all_texts = []
    all_x, all_y = [], []
    for eps in EPSILON_ORDER:
        sub = [r for r in agg if r["epsilon"] == eps]
        if not sub:
            continue
        # PGD grid cells (steps/alpha set) get the dashed connector + std
        # band within this epsilon. Noise-control cells (steps=None) have
        # only one point per epsilon, so they're handled separately below as
        # their own cross-epsilon series (connected + shaded across the 3
        # epsilons instead of within one).
        pgd_sub = sorted((r for r in sub if r["steps"] is not None),
                         key=lambda r: r[x_mean_key])
        color = EPSILON_COLORS.get(eps, DEFAULT_COLOR)

        if pgd_sub:
            xs  = [r[x_mean_key] for r in pgd_sub]
            ys  = [r[y_mean_key] for r in pgd_sub]
            std = [r[y_std_key]  for r in pgd_sub]
            ax.plot(xs, ys, linestyle="--", color=color, linewidth=1 * scale, zorder=1)
            if not args.no_std_band:
                ax.fill_between(
                    xs,
                    [y - 0.5 * s for y, s in zip(ys, std)],
                    [y + 0.5 * s for y, s in zip(ys, std)],
                    color=color, alpha=0.15, zorder=0,
                )
            all_x.extend(xs)
            all_y.extend(ys)

        for row in pgd_sub:
            if args.errorbars:
                ax.errorbar(
                    row[x_mean_key], row[y_mean_key],
                    xerr=row[x_std_key], yerr=row[y_std_key],
                    fmt="o", markersize=6.5 * scale, color=color,
                    ecolor=color, elinewidth=0.8 * scale, capsize=2, zorder=3,
                )
            else:
                ax.plot(row[x_mean_key], row[y_mean_key],
                        "o", markersize=6.5 * scale, color=color, zorder=3)

            if args.annotate:
                t = ax.text(
                    row[x_mean_key], row[y_mean_key],
                    f"{row['steps']}/{row['alpha']:.2f}",
                    fontsize=9 * text_scale, color=color, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.75),
                )
                all_texts.append(t)

        ax.plot([], [], "o--", color=color, markersize=6.5 * scale, linewidth=1 * scale,
                label=rf"$\varepsilon$={eps}")

    # Gaussian-noise control: one point per epsilon, connected across
    # epsilons (not within one, like the PGD lines above) since that's the
    # only axis along which noise varies — this gives fill_between an actual
    # x-span to shade, unlike a lone per-epsilon point.
    noise_sub = sorted((r for r in agg if r["steps"] is None),
                       key=lambda r: r[x_mean_key])
    if noise_sub:
        xs  = [r[x_mean_key] for r in noise_sub]
        ys  = [r[y_mean_key] for r in noise_sub]
        std = [r[y_std_key]  for r in noise_sub]
        ax.plot(xs, ys, linestyle="--", color=DEFAULT_COLOR, linewidth=1 * scale, zorder=1)
        if not args.no_std_band:
            ax.fill_between(
                xs,
                [y - 0.5 * s for y, s in zip(ys, std)],
                [y + 0.5 * s for y, s in zip(ys, std)],
                color=DEFAULT_COLOR, alpha=0.15, zorder=0,
            )
        all_x.extend(xs)
        all_y.extend(ys)

        for row in noise_sub:
            color = EPSILON_COLORS.get(row["epsilon"], DEFAULT_COLOR)
            if args.errorbars:
                ax.errorbar(
                    row[x_mean_key], row[y_mean_key],
                    xerr=row[x_std_key], yerr=row[y_std_key],
                    fmt="*", markersize=10 * scale, color=color, markeredgecolor="black",
                    markeredgewidth=0.5 * scale, ecolor=color, elinewidth=0.8 * scale, capsize=3, zorder=3,
                )
            else:
                ax.plot(row[x_mean_key], row[y_mean_key],
                        "*", markersize=10 * scale, color=color,
                        markeredgecolor="black", markeredgewidth=0.5 * scale, zorder=3)

            if args.annotate:
                t = ax.text(
                    row[x_mean_key], row[y_mean_key], "noise",
                    fontsize=9 * text_scale, color=color, zorder=5, style="italic",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.75),
                )
                all_texts.append(t)

        ax.plot([], [], "*--", color=DEFAULT_COLOR, markersize=10 * scale,
                markeredgecolor="black", markeredgewidth=0.5 * scale, linewidth=1 * scale,
                label="gaussian noise")

    if args.annotate and all_texts:
        # Repel labels from each other AND from every marker (across all
        # epsilon series at once) so nearby points from different series
        # don't collide; draw thin leader lines where labels had to move.
        adjust_text(
            all_texts,
            x=all_x, y=all_y,
            expand_text=(1.6, 2.0),
            expand_points=(2.2, 2.4),
            force_text=(1.0, 1.2),
            force_points=(0.6, 0.9),
            lim=2000,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4 * scale, alpha=0.7,
                             shrinkA=1, shrinkB=3),
            ax=ax,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(*y_lim)
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if invert_x:
        ax.invert_xaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2)
    if legend:
        ax.legend(frameon=False)
    return True


# Axis settings per x metric, shared by the single and side-by-side figures.
X_METRICS = {
    "lpips": dict(x_key="lpips", x_label="LPIPS", x_lim=(0.05, 0.5)),
    "ssim":  dict(x_key="ssim",  x_label="SSIM",  x_lim=None, invert_x=True),
}


def plot(agg: list, args: argparse.Namespace, output_stem: str, **kw) -> None:
    """One standalone figure; kw forwarded to draw()."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    if draw(ax, agg, args, **kw):
        save(fig, output_stem)
    plt.close(fig)


def plot_asr_vs_config(rows: list, args: argparse.Namespace, output_stem: str) -> None:
    """ASR vs PGD configuration: x = the (steps, alpha) grid, grouped by step
    size alpha with the step count as tick label; one line per epsilon (same
    colours as the LPIPS/SSIM plots). Each point = mean ASR over all
    (seed, held_out) rows of that cell; shaded band = ±1 std over those same
    rows (random starts × leave-one-out models). Gaussian noise has no
    steps/alpha, so it's a dashed horizontal line with a star at every x
    per epsilon (no band; its
    std is printed). The y range is widened to fit every PGD band."""
    configs = [(st, a) for a in ALL_STEP_SIZES for st in ALL_STEP_COUNTS]
    fig, ax = plt.subplots(figsize=(8, 6))

    print("\nASR vs (steps, alpha) — mean over seeds × held-out")
    print(f"{'epsilon':<8} {'steps':>5} {'alpha':>6} {'ASR':>7} {'±':>7} {'n':>4}")
    y_lo, y_hi = 0.0, 1.0
    for eps in EPSILON_ORDER:
        color = EPSILON_COLORS.get(eps, DEFAULT_COLOR)
        xs, ys, sds = [], [], []
        for i, (st, a) in enumerate(configs):
            vals = [r["asr"] for r in rows if r["epsilon"] == eps and r["steps"] == st
                    and r["alpha"] == a and r["asr"] is not None]
            if len(vals) < max(args.min_runs, 1):
                continue
            xs.append(i)
            ys.append(float(np.mean(vals)))
            sds.append(float(np.std(vals)))
            print(f"{eps:<8} {st:>5} {a:>6.2f} {ys[-1]:>7.4f} {sds[-1]:>7.4f} {len(vals):>4}")
        noise = [r["asr"] for r in rows if r["epsilon"] == eps and r["steps"] is None
                 and r["asr"] is not None]
        if not xs and not noise:
            continue
        ax.plot(xs, ys, "o-", color=color, markersize=8, linewidth=2, zorder=3,
                label=rf"$\varepsilon$={eps}")
        if xs and not args.no_std_band:
            lo, hi = np.subtract(ys, sds), np.add(ys, sds)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=1)
            y_lo, y_hi = min(y_lo, lo.min()), max(y_hi, hi.max())
        if noise:
            mu, sd = float(np.mean(noise)), float(np.std(noise))
            ax.plot(range(len(configs)), [mu] * len(configs), "*--", color=color,
                    markersize=11, markeredgecolor="black", markeredgewidth=0.5,
                    linewidth=1.5, zorder=2)
            print(f"{eps:<8} {'noise':>5} {'-':>6} {mu:>7.4f} {sd:>7.4f} {len(noise):>4}")
    ax.plot([], [], "*--", color=DEFAULT_COLOR, markersize=11, markeredgecolor="black",
            markeredgewidth=0.5, linewidth=1.5, label="gaussian noise")

    # Step count as tick label; step size as group label under each block of
    # len(ALL_STEP_COUNTS) ticks, blocks separated by thin vertical lines.
    n_st = len(ALL_STEP_COUNTS)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([str(st) for st, _ in configs])
    for g, a in enumerate(ALL_STEP_SIZES):
        ax.annotate(rf"$\alpha$={a}", xy=(g * n_st + (n_st - 1) / 2, 0),
                    xycoords=("data", "axes fraction"), xytext=(0, -30),
                    textcoords="offset points", ha="center", va="top")
        if g:
            ax.axvline(g * n_st - 0.5, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xlim(-0.5, len(configs) - 0.5)
    ax.set_xlabel("PGD steps", labelpad=34)
    ax.set_ylabel("ASR")
    ax.set_ylim(np.floor(y_lo * 10) / 10, np.ceil(y_hi * 10) / 10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3)
    # errorbar containers sort after plain lines in the legend — put noise last.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] == "gaussian noise")
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=len(labels), handletextpad=0.3, columnspacing=1.0)
    save(fig, output_stem)
    plt.close(fig)


def save(fig, output_stem: str, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


# ─────────────────────────── arg parsing ─────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot AUROC/ASR vs LPIPS and/or SSIM for the PGD ablation grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir", default=EVAL_OUTPUT_DIR,
                   help="Root of the git-tracked evaluate_detector results tree "
                        "(default: $EVAL_OUTPUT_DIR).")
    p.add_argument("--clean_dir", default=CLEAN_DIR,
                   help="Dir with clean-baseline results.json/<model>/, used for ASR.")
    p.add_argument("--epsilons", nargs="+", default=None)
    p.add_argument("--seeds",    type=int, nargs="+", default=None)
    p.add_argument("--min_runs", type=int, default=0,
                   help="Drop triples with fewer rows.")
    p.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--x_metric", default="lpips", choices=["lpips", "ssim", "config"],
                   help="x axis. lpips/ssim: AUROC + ASR scatter per ablation "
                        "cell; config: ASR vs (steps, alpha), one line per epsilon.")
    p.add_argument("--errorbars",  action="store_true")
    p.add_argument("--no_std_band", action="store_true",
                   help="Don't draw the shaded std-dev band around each line.")
    p.add_argument("--annotate",     dest="annotate", action="store_true",  default=True)
    p.add_argument("--no-annotate",  dest="annotate", action="store_false")
    p.add_argument("--output_auroc", default=str(Path(OUTPUT_DIR) / "auroc_vs_lpips"),
                   help="Output path stem for the AUROC-vs-LPIPS figure "
                        "(no extension; .png and .pdf written).")
    p.add_argument("--output_asr", default=str(Path(OUTPUT_DIR) / "asr_vs_lpips"),
                   help="Output path stem for the ASR-vs-LPIPS figure "
                        "(no extension; .png and .pdf written).")
    p.add_argument("--output_auroc_ssim", default=str(Path(OUTPUT_DIR) / "auroc_vs_ssim"),
                   help="Output path stem for the AUROC-vs-SSIM figure.")
    p.add_argument("--output_asr_ssim", default=str(Path(OUTPUT_DIR) / "asr_vs_ssim"),
                   help="Output path stem for the ASR-vs-SSIM figure.")
    p.add_argument("--output_asr_config", default=str(Path(OUTPUT_DIR) / "asr_vs_config"),
                   help="Output path stem for the ASR-vs-(steps, alpha) figure "
                        "(--x_metric config).")
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_data(args.results_dir, args.lpips_net, args.clean_dir,
                     args.epsilons, args.seeds)
    if not rows:
        print(f"ERROR: no data found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    if args.x_metric == "config":
        plot_asr_vs_config(rows, args, args.output_asr_config)
        return

    if args.x_metric == "ssim":
        # Restrict to rows with an ssim.json so SSIM, AUROC and ASR are all
        # averaged over the same (seed, held_out) cells.
        n_all = len(rows)
        rows = [r for r in rows if r["ssim"] is not None]
        if len(rows) < n_all:
            print(f"WARNING: {n_all - len(rows)}/{n_all} rows have no ssim.json — "
                  f"excluded from SSIM plots.")
    agg = aggregate(rows, args.min_runs) if rows else []
    if not agg:
        print("ERROR: no triples survived aggregation.", file=sys.stderr)
        sys.exit(1)
    outs = ((args.output_auroc, args.output_asr) if args.x_metric == "lpips"
            else (args.output_auroc_ssim, args.output_asr_ssim))
    plot(agg, args, outs[0], y_key="auroc", y_label="AUROC", y_lim=(0.0, 0.65),
         **X_METRICS[args.x_metric])
    plot(agg, args, outs[1], y_key="asr", y_label="ASR", y_lim=(0.0, 1.0),
         **X_METRICS[args.x_metric])

if __name__ == "__main__":
    main()
