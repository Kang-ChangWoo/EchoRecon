# Pre-registered criteria — E100 cause decomposition of the falling view-count curve

Written 2026-09-20 (session 2) before any E100 result was computed, committed
so the hash proves it. Not changed after results are read. Numbers already
seen before this file: Stage A (E01), Stage B, Stage D v3 (`results/E30_stage_d/`).

## Question

F1@0.2 of the fused point cloud peaks at N=4 views and falls afterwards
(Stage A / Stage D v3, r2 and r8). Five candidate causes are separated:

| id | cause | how it is removed in isolation | needs training |
|---|---|---|---|
| e | reference grows with N (GT of the *selected* N steps) | score the same predictions against a fixed, N-independent reference = GT of **all** steps of the sequence | no |
| a | front-end time resolution (STFT win 400 = 8.33 ms = 1.43 m depth) | retrain the base model with a shorter window, same hop, everything else equal; redo predictions and the curve | yes |
| b | no angular diversity (straight trajectory, fixed heading) | within a sequence, at the same N, compare the most spread subset with the most compact (consecutive) subset; and r2 vs r8 (per-observation heading diversity) | no |
| c | ray-independence assumption | measure signed-error correlation between rays of one view on the same GT plane, and across views at the same surface voxel; control = plane-wise (edge-preserving) smoothing of each view's depth before fusion, nothing else changed | no |
| d | no correspondence check (free-space contradiction) | remove a voxel proposed by view i when other views' rays passed through it (their predicted depth lies beyond the voxel by a margin); nothing else changed | no |

## Fixed pipeline (the "baseline" whose curve is being explained)

`A_point`: base-model point depth (`outputs/pred/<mode>/`), per-face -> radial,
unproject at pixel stride 2, 0.1 m voxels, voxel position = plain mean.
Two rankings are recorded but the **primary** number is `frac = 1.0` (every
voxel, no ranking), because a ranking would mix a sixth cause (support) into
the measurement. Secondary: support top 25 %. N in {1, 2, 4, 8, 16, all},
subset seed 0 = evenly spaced (as in Stage A / D), seeds 1-2 random, for the
subset spread. Test scenes {apartment_2, frl_apartment_5, office_4} (39
sequences) give every reported number; the val scenes (37 sequences) choose
hyper-parameters only. Both `r2` and `r8`.

## Definitions

- **drop16** = mean over test sequences of [F1@0.2(N=4) − F1@0.2(N=16)],
  paired per sequence, seed 0, frac 1.0. Sequences shorter than 16 steps use
  all their steps at N=16 (as in Stage A; `N` column records the true count).
  Also recorded: **dropAll** = F1(N=4) − F1(all).
- **recovery(X)** = drop16(baseline) − drop16(X removed), on the same
  sequences and the same reference. Positive = the curve falls less.
- **curve flipped** = F1(N=16) ≥ F1(N=4) − 0.01 **and** F1(all) ≥ F1(N=4) − 0.01
  (i.e. more views no longer hurt).
- **Tie band 0.03 F1** (Stage A subset-seed spread, same as `E30_stage_d/CRITERIA.md`).
  A recovery is *meaningful* if recovery ≥ 0.03 **and** the paired bootstrap
  95 % CI over test sequences (10 000 resamples) excludes 0, **on both r2 and r8**.
  If only one of r2/r8 passes: "partial". If |recovery| < 0.03: "no effect".
- Once (e) is decided, every later number is reported under **both**
  references (subset, fixed-all); the verdict uses the fixed-all reference if
  (e) is meaningful, the subset reference otherwise.

## "The cause is real" (existence) thresholds

| cause | measurement | real if |
|---|---|---|
| e | n_ref(N=16) / n_ref(N=4) | ≥ 1.2 (trivially expected; the *effect* is judged by recovery) |
| a | single-view ERP MAE on test, short window vs win 400 (same hop) | improves by ≥ 0.02 m |
| b | F1@0.2 at N=8: evenly-spread subset − consecutive subset (same sequence) | ≥ 0.03; and voxel precision@0.2 in the top tertile of viewing-angle spread exceeds the bottom tertile by ≥ 0.05 at matched support |
| c | Pearson r of signed radial error between same-view ray pairs on the same GT plane, 5–10° apart | ≥ 0.5 and ≥ 0.2 above different-plane pairs at the same separation; cross-view: r between two views' signed errors at the same GT surface voxel ≥ 0.3 means views are not independent evidence |
| d | wrong rate (nearest-reference distance > 0.2 m) of contradicted voxels − wrong rate of uncontradicted voxels | ≥ 0.15 absolute |

## Removal variants and hyper-parameter selection (val only)

- (a) windows {400 (released, hop 160), 400, 128, 64} with hop 32 for the
  three retrained ones (hop ≤ win/2 must hold for every window; the released
  hop 160 would skip signal at win 64). Same recipe as `oaa_r2_fin`
  (40 ep, bs 24, lr 5e-4, seed 0, n_fft 512 zero-padded so the frequency
  bin count is unchanged). r2 only unless time allows r8. Nothing is
  selected: all windows are reported.
- (b) subsets: "spread" = `linspace` (seed 0), "compact" = N consecutive
  steps centred in the sequence. No hyper-parameter.
- (c) smoothing: median filter on the per-view ERP depth with window
  {5, 9, 15} px, selected on val by drop16 recovery; the selected value is
  read on test. Also recorded: the same filter's effect on single-view MAE
  (if MAE gets worse and the curve improves, that is reported as such).
- (d) contradiction rule: voxel with support s and contradiction count c is
  dropped if c ≥ k·s for k in {0 (c ≥ 1), 1, 2}; margin 0.2 m along the
  contradicting ray; selected on val. Free space of view j = voxels on its
  rays from the origin to (predicted depth − margin).

## Interactions

(c) and (d) are measured on the released model first, then again on the
short-window model of (a) if it exists. If a recovery changes by more than
0.03 between the two, the interaction is reported as a result in its own right.

## Tie-breaking

If two or more causes are meaningful and within 0.03 of each other, the one
that is meaningful under both references and both observation sets wins; if
still tied, the cheaper one (no retraining) is chosen for the final method.
If none is meaningful, the result is "the fall is not explained by these
five", reported with the numbers.

## Not criteria

Val F1 never ranks anything. No new oracle. Metrics at 0.1 m are recorded but
not used for the verdict.
