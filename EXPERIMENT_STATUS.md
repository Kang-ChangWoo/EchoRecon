# Experiment status

Audit of what exists in this repository, judged against the standard that an
experiment counts as DONE only with code, a run log, a result file, the scenes
and seeds it used, an evaluation script, and a result reproducible at the
current revision. Updated as work proceeds.

## Conventions this repository already fixes (verified, not assumed)

| item | value | how it was verified |
|---|---|---|
| echo input | binaural, 48 kHz, cropped to the 10 m round trip (2799 samples), magnitude STFT n_fft 512 / win 400 / hop 160, first 256 bins, nearest-resized to 256x512 | read from the base model's data module; `predict.py` reproduces it |
| observation sets | r2 = one heading (2 ch), fb = front+back (4), r6 = three headings (6), r8 = four (8); channel order [000 L,R \| 090 \| 180 \| 270] | `predict.py:CHAN`, matched against each checkpoint's `nviews` |
| ERP | 256x512 output, 512x1024 ground truth; centre column = forward, azimuth increasing to the right ("right0") | `convention_check.py` (consecutive-frame agreement) and `map_check.py` (94.8 % of wall-height points within 10 cm of the floor plan) |
| ERP inverse | direction -> pixel round-trips exactly on 2000 random pixels | unit check in this session; lives in `support_diag.dir_to_pixel` |
| depth meaning | the model predicts per-face cubemap z-depth, not radial: `erp_depth = erp_depth_radial * max(\|dx\|,\|dy\|,\|dz\|)` | recovered radial matches `erp_depth_radial` to 0.0000 m, corr 1.00000 |
| depth range | 0.1 to 10 m, clipped at 10 | checkpoint args (`max_depth`) |
| pose | habitat position xyz (y up) + quaternion (w,x,y,z); known, never estimated | `data.Sequence.pose`, used only in unprojection |
| split | val {apartment_1, frl_apartment_4, office_3}, test {apartment_2, frl_apartment_5, office_4}; 39 test sequences, 14-60 steps each, 0.15 m apart | `data.py`, follows the base model's scene-disjoint split |
| voxel / tau | 0.1 m voxels, tau 0.2 m, pixel stride 2 | `eval_fusion.py` defaults, recorded in every output's metadata |

## Status table

| ID | Experiment | Status | Existing evidence | Missing | Result | Next action | Commit |
|---|---|---|---|---|---|---|---|
| E00 | single-view point depth | DONE | `outputs/summary_*.json` plus `results/E01_view_curve/r2/per_scene.csv` (N=1 row) | | ERP MAE 0.296 (r2) to 0.253 (r8); single-step accuracy 0.44 -> 0.40 m | keep | 328d5c9 |
| E01 | view-count curve | DONE | `results/E01_view_curve/r2/` (8892 rows, N = 1/2/4/8/16/32/all, 3 seeds, full metric set) | other observation sets | Chamfer, F1 and IoU peak at N=4 and decline after; accuracy degrades, completeness improves | keep | Stage A |
| E02 | several scenes and seeds | DONE | per-scene and per-seed CSV in the same folder | | all 3 scenes and 3 seeds show the same shape; 87 % of sequence-seed pairs peak at N<=4 | keep | Stage A |
| E03 | uniform fusion degradation | DONE | `summary_*.json` | | fused-all 0.60 m against single-step 0.44 m (r2) | keep | 328d5c9 |
| E04 | voxel support filtering | DONE | kept-fraction curves in `summary_*.json` | | top 25 % by support 0.29 m, completeness 0.25 -> 0.54 m | keep | 328d5c9 |
| E05 | support threshold sweep | DONE | kept fractions 1.0/0.9/0.75/0.5/0.25/0.1 for both rankings | absolute thresholds | support top 25 % F1 peaks at N=4 (0.594) | keep | Stage A |
| E06 | accuracy-completeness Pareto | DONE | the same six fractions give the curve | | | keep | Stage A |
| E07 | oracle headroom | DONE | `fused.oracle` and the ranked curves | | oracle top-quarter F1 rises with every view, 0.660 -> 0.855, while support peaks at N=4 | keep | Stage A |
| E10-E15 | robust point fusion | DONE | `results/E10_robust_point/r2/` (3432 rows, 10 families swept, N = 4 and all) | a per-ray consensus (the per-voxel one is inert) | best point-only closes 24 % of the oracle gap at every view, 27 % at N=4; no robust operation beats counting support | STOP CHECK A does not fire | Stage B |
| E20-E25 | depth posterior | DONE | `outputs/posterior/posterior_r2/`, `results/E21_posterior_rays/posterior_r2_test.json`, `results/E25_fake_posterior/r2/` | r8 head still training; K = 64/256 not swept | argmax MAE 0.246 m (point baseline 0.289); mode rescue@10 8.6 % against the fake Gaussian's 0.0 %; 1.28 modes/ray; fake posterior fusion equals support counting on matched candidates | Stage D | Stage C |
| E30-E32 | same-backbone comparison | DONE | `results/E30_stage_d/{r2,r8}_v2/` (+ `_v2_val`; 10800 rows each; band interpolated, T applied, kept fraction chosen on val) | fb/r6; several view subsets | point and argmax agree at N=1 (0.572 / 0.576 F1); full posterior falls with N exactly as they do and is worse from N=4 on | CASE 4: hypothesis not supported | Stage D |
| E40-E43 | posterior fusion | PARTIAL | soft surface consensus on two candidate sets, band interpolation fixed, temperature applied, confidence-weighted and confidence-filtered variants | tau and bin sensitivity (E42), truncation study (E43) | posterior fusion beats the point pipeline only at N=1-2 and by <0.02; confidence weighting ±0.01; confidence filtering -0.14 to -0.30 | sweep tau/bins only if the direction continues | Stage D v2 |
| E50-E51 | free-space ablation | DONE as E100 (d) | `results/E100_cause/{r2,r8}/` contra0/1/2 and support_match rows | | contra1 recovers 0.036 / 0.039 and lifts F1(all) by 0.05, but the budget-matched support ranking does better (−0.039 / −0.030 for contra1 at N=all): no information beyond support | E100 | see git log |
| E60-E62 | view diversity, shuffle, error correlation | DONE as E100 (b)(c) | `results/E100_cause/{r2,r8}/measurements.json`, compact rows | pose shuffle | spread beats consecutive subsets by 0.06 (N=4) to 0.02 (N=16); cross-view signed-error r 0.9 at <0.5 m baseline, 0.4 at >=1.5 m; within-view same-plane r 0.93 at 5–10°; plane smoothing recovers nothing | E100 | 8a8007f |
| E70-E71 | learned fusion | NOT_DONE | | | | Stage G | |
| E100 | cause decomposition of the falling curve (e)(a)(b)(c)(d) | PARTIAL | `results/E100_cause/` (CRITERIA.md pre-registered 69102f2; r2, r8, r2_val, r8_val) | | (e) real: 0.022 of the all-voxel fall, and the *whole* fall of the support-top-25 % curve, which rises monotonically to N=16 and saturates under a fixed reference (drop16 −0.048 / −0.054); precision falls with N at every operating point; (c)(d) not distinct causes, (b) the mechanism of the precision decline; (a) short windows (128/64 at hop 32) and the hop-only control (400/32) change nothing on r2 (single-view MAE −0.003/−0.007/−0.003, drop16 +0.041/+0.032/+0.036); r8 win 128 is worse single-view (+0.013) with the same curve (drop16 +0.030) | close; next question is the 0.56 saturation | 0ea8ea1 |
| E101 | marginal-view analysis (why precision falls with N) | DONE | `results/E101_marginal_view/{r2,r8}/` (CRITERIA_E101.md pre-registered 0b6e7f4) | val split, short-window models | added view: overlapping points .7–.85 precise, new voxels .2–.5, worse when < 0.5 m from an included view (Spearman +0.4); wrong remainder split 40/30/30 % near/displaced/hallucinated | E102 baseline-aware support | see git log |
| E102 | baseline-aware support (Δ-cells, views, span vs point count) | DONE, negative | `results/E102_baseline_support/{r2,r8}/` (CRITERIA_E102.md c135307) | | no ranking beats point count by ≥ 0.03 (views +0.004/+0.006; Δ ≥ 1 m worse, −0.02); geometry of the fused cloud is exhausted | E103: per-ray confidence under the fixed reference | see git log |
| E103 | per-ray confidence rankings under the fixed reference | DONE, negative | `results/E103_confidence/{r2,r8}_post/` (CRITERIA_E103.md 1321bef) | conf-vs-support rank correlation | conf_sum −0.005/−0.008 vs support, conf_mean −0.05, median filter loses to support at equal count; confidence is redundant with support at the voxel level | owner decision: data geometry or predictor | see git log |
| E104 | lobe / region each observation does well | DONE, negative | `results/E104_lobe/` | solid-angle-corrected elevation curve | azimuth wrong-rate spread 0.13 (r2) / 0.06 (r8) < 0.15: no lobe; elevation spread 0.6 but shared by all views; elevation weighting −0.002 vs support | closed | see git log |
| E105 | posterior fusion under the fixed reference (Stage D v3 --ref fixed) | DONE, negative | `results/E30_stage_d/{r2,r8}_v3fixed/` | | C below A by 0.04–0.06 at N=all on both sets; C still falls with N while A does not | closed | see git log |
| E102b | distance-based view selection (issue B2) | DONE, negative | `results/E102b_view_select/{r2,r8}/` (CRITERIA_B_C.md) | | Δ-spaced vs random subset of the same count: +0.009…+0.021 at frac 1.0 (band 0.03); the all-voxel fall is a function of count, not spacing | closed | see git log |
| E107 | confidence paradox measured (issue C) | DONE | `results/E107_conf_paradox/` | | median-confidence filter deletes 100 % of correct voxels beyond 4 m, 93–95 % of oblique, 85–98 % of mid-range; recall beyond 4 m → 0.000; (ii) supported 2.5–3×, (i) 0.47 < 0.6 (loss is broader than far/oblique) | headline evidence for representing rather than discarding uncertainty | see git log |
| E106 | learned per-voxel evidence (graph-premise proxy) | DONE | `results/E106_learned/` (CRITERIA_E104_E106.md) | learner on train-scene predictions; interaction with the retrained head | MLP ranking +0.037/+0.037 over count (meaningful); single-view group +0.031/+0.021, inter-view group +0.018/+0.023 (band): graph premise not supported, learned single-view prior is | do not build the bipartite model | see git log |
| A-retrain | posterior head, bin-wise warm start, retrained; head-only variant (issue A) | DONE | `outputs/posterior/posterior_{r2,r8}_warm/`, `results/E21_posterior_rays/*_warm_test.json`, `results/E30_stage_d/*_warm_v3*/` | | warm: best epoch 4 / 2, Stage D within 0.01 of the flat head; head-only: best epoch 8 / 12, 0.02–0.04 worse; C below A by 0.04–0.07 at N=all with all three heads → limitation confirmed | closed | see git log |
| E110 | top-k candidates per ray (flat heads; warm/head-only passes stored) | DONE, negative | `results/E110_topk/{r2,r8}/`, `outputs/pred/*_post_topk` | warm/head-only fusion | val picks k=1; at matched budget top-k never exceeds B_argmax (mass k=2: .546/.578 vs .554/.573); far recall rises with k only with a steeper precision loss | closed | see git log |
| E108 | expected-count normalisation (pred / GT visibility, smoothed density) | DONE, negative | `results/E108_E109_continuity/{r2,r8}/` | | far recall .05→.42 but F1 −0.08…−0.10 (GT-visibility bound −0.18…−0.21); thresholded vs same-count −0.036; smoothed threshold +0.000 | closed: the far field is wrong, not under-voted | see git log |
| E109 | continuity-aware accumulation (a/b/c, measured rho and exp-lambda) | DONE, negative | same | | all within ±0.014 of count at N=all, both refs; slope not steepened; rho form irrelevant | closed | see git log |
| E106b | learned prior fitted on the train scenes (val for hyper-parameters), transfer and refit on the retrained heads | DONE | `results/E106b_trainfit/*/summary.txt` | per-ray version inside the 2-D model (design change) | +0.050 / +0.054 over count (F1 .605 / .627); transfer +0.03…+0.06; refit up to +0.088 (head-only) | owner decision | see git log |
| E111 | per-ray reliability head in the 2-D model (head-only, joint; r2, r8) | DONE, negative | `results/E111_reliability/`, `results/E106b_trainfit/*_rel_*`, ckpts `/root/local1/changwoo/echorecon_ckpt/` | multi-view head (design change) | Σr ranking −0.02 vs count, ranker +0.059/+0.043, ranker+head +0.001; far bin below the E110 curve; joint best = warm-up epoch | owner decision | see git log |
| E80-E82 | room-scale, observed-space, DAPS-compatible evaluation | NOT_DONE | current reference is the fused ground-truth depth of the same steps, which is a diagnostic, not a mesh ground truth | mesh occupancy, IoU/Chamfer/NC/F-score, DAPS protocol | | Stage H | |
| E90-E93 | pose noise, SNR, materials | NOT_DONE | | | out of scope for now per `DIRECTION.md` | | |

## Earlier work in this repository, by the same standard

| work | status | note |
|---|---|---|
| ERP convention and floor-plan check | DONE | `convention_check.py`, `map_check.py` |
| per-face depth fix | DONE | changed every downstream number; everything before commit c37e2b6 is INVALID |
| blur diagnosis (step 2) | DONE | `blur_diag.py`, `outputs/blur_*.json`; prediction is as rough as the truth, not an over-smoothed shell |
| hop sweep | INVALID as an answer to its question | hops 40/80/320 are off-distribution for a model trained at 160; needs retraining per hop |
| three-state agreement (step 3) | DONE | `support_diag.py`; every agreement score ranks worse than counting points |

## Known gaps in the evaluation itself

- The full metric set (accuracy, completeness, symmetric Chamfer, precision /
  recall / F1 at three thresholds, voxel IoU, coverage) is implemented in
  `src/fuse.py:metrics` and used from Stage A on. Results predating it report
  accuracy and completeness only.
- The reference is the ground-truth depth of the same steps, fused. It
  therefore cannot penalise a method for missing what no view could see, and it
  is not comparable to DAPS. A mesh-derived occupancy is required for the
  headline table.
- View subsets are evenly spaced with one deterministic subset per N in Stages B-D (Stage A: 3 seeds), so no variance over subset
  choice is available.
