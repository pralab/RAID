# RAID whitebox / leave-one-out attack runtimes

| Stage | Start | End | Duration (h:m:s) |
|---|---|---|---|
| White-box (eps=16/255) | 2026-09-04 19:36:03 | 2026-09-05 00:00:12 | 04:24:09 |
| Leave-one-out (eps=16/255) | 2026-09-05 02:52:14 | 2026-09-05 17:20:16 | 14:28:02 |
| White-box (eps=32/255) | 2026-09-05 17:20:16 | 2026-09-05 20:15:42 | 02:55:26 |
| Leave-one-out (eps=32/255) | 2026-09-05 20:15:42 | 2026-09-06 10:44:10 | 14:28:28 |
| Full ensemble (7-model, full dataset, eps=8/16/32-255) | 2026-09-06 10:44:42 | 2026-09-08 01:21:00 | 38:36:18 |

Each white-box stage = 7 single-model runs at subset=1000. Each leave-one-out
stage = 7 six-model-ensemble runs at subset=1000.

## Subsampling mode runtime estimate (inferred, not measured)

Subsampling mode runs 7 white-box + 7 leave-one-out attacks per seed, at
subset=200 (5x smaller than the 1000-sample runs above), across 5 seeds.

Extrapolated by scaling the measured 1000-sample runtimes linearly by
200/1000 = 0.2x. This assumes runtime scales linearly with sample count,
which **ignores fixed per-run overhead** (model loading, checkpointing) -
real 200-sample runs will likely finish faster than this linear estimate,
since that overhead is the same regardless of subset size. Treat these as
upper-bound estimates, not measurements.

| Epsilon | White-box (7 models, 200 imgs, scaled) | LOO (7 ensembles, 200 imgs, scaled) | Per seed | x5 seeds |
|---|---|---|---|---|
| 16/255 | 04:24:09 × 0.2 ≈ 0:52:50 | 14:28:02 × 0.2 ≈ 2:53:36 | ≈ 3:46:26 | ≈ 18h 52m |
| 32/255 | 02:55:26 × 0.2 ≈ 0:35:05 | 14:28:28 × 0.2 ≈ 2:53:42 | ≈ 3:28:47 | ≈ 17h 24m |
| 8/255 | no measured white-box/LOO runtime available | no measured white-box/LOO runtime available | - | - |

No isolated white-box/LOO timing exists for eps=8/255 - only the combined
38:36:18 full-ensemble figure spans all three epsilons together, so it
can't be decomposed into a per-epsilon white-box/LOO estimate.

**Bottom line:** ~17-19h per epsilon budget (16/255, 32/255) for the full
5-seed subsampling run (70 attack runs each), by linear extrapolation from
measured 1000-sample runtimes. Actual runtime is plausibly somewhat lower
due to fixed overhead not scaling with subset size - no measured data to
pin down that gap yet.

## Additional mode (APGD + CWA leave-one-out, 1000 images)

14 runs per epsilon (7 LOO × 2 attacks), run in parallel across 3 GPUs
(one epsilon per GPU). Timestamps from adv_dataset.pkl file modification
times.

| Epsilon | GPU | First output | Last output | Wall clock |
|---|---|---|---|---|
| 8/255  | cuda:0 | 2026-09-19 16:17 | 2026-09-20 20:55 | ~28h 38m |
| 16/255 | cuda:1 | 2026-09-19 16:17 | 2026-09-20 19:58 | ~27h 41m |
| 32/255 | cuda:2 | 2026-09-19 16:17 | 2026-09-20 20:04 | ~27h 47m |

Total: 42 runs (all complete). Note: wall clock includes time lost to the
initial OOM crash and restart with the fixed APGD code; steady-state runtime
per run is likely ~1h 50m (28h / 14 runs ≈ 2h per run including overhead).
