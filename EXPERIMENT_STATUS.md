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
| E00 | single-view point depth | DONE | `outputs/summary_*.json` (`erp_mae`, `single`), 39 sequences x 4 observation sets | per-scene CSV | ERP MAE 0.296 (r2) to 0.253 (r8); single-step accuracy 0.44 -> 0.40 m | keep | 328d5c9 |
| E01 | view-count curve | PARTIAL | N = 2, 4, 8, 16, all for r2 and r8 (`summary_*_N*.json`) | N = 1 and 32; precision/recall/F1/IoU/Chamfer; no seed variation | fused accuracy degrades monotonically 0.48 -> 0.60 m; completeness flat from N = 4 | extend | |
| E02 | several scenes and seeds | PARTIAL | 3 test scenes, 39 sequences | view subsets are evenly spaced and deterministic; no seed | | add seeded subsets | |
| E03 | uniform fusion degradation | DONE | `summary_*.json` | | fused-all 0.60 m against single-step 0.44 m (r2) | keep | 328d5c9 |
| E04 | voxel support filtering | DONE | kept-fraction curves in `summary_*.json` | | top 25 % by support 0.29 m, completeness 0.25 -> 0.54 m | keep | 328d5c9 |
| E05 | support threshold sweep | PARTIAL | four fractions 0.75/0.5/0.25/0.1 | a dense sweep and absolute thresholds | | extend | |
| E06 | accuracy-completeness Pareto | PARTIAL | four points per method | full curve, precision/recall, F1 | | extend | |
| E07 | oracle headroom | DONE | `fused.oracle` in every summary | | oracle top 25 % 0.076 m / 97 % within 0.2 m at completeness 0.275 m | keep | a461855 |
| E10-E15 | robust point fusion | NOT_DONE | TSDF exists but is not evaluated as a baseline | outlier removal, clustering, trimmed consensus, keyframes | | Stage B | |
| E20-E25 | depth posterior | NOT_DONE | | head, training, calibration, rescue@K, fake Gaussian | | Stage C | |
| E30-E32 | same-backbone comparison | NOT_DONE | | | | Stage D | |
| E40-E43 | posterior fusion | NOT_DONE | | | | Stage D | |
| E50-E51 | free-space ablation | NOT_DONE | | | | Stage E | |
| E60-E62 | view diversity, shuffle, error correlation | PARTIAL | `support_diag.py` measured delta-restricted support (a diversity proxy) | pose shuffle, duplicate vs diverse views, error correlation against translation/yaw/overlap | delta restriction degrades every score; no evidence of neighbour-shared bias | Stage F | 328d5c9 |
| E70-E71 | learned fusion | NOT_DONE | | | | Stage G | |
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

- Metrics reported so far are accuracy (mean nearest distance and fraction
  within tau) and completeness. Precision, recall, F1/F-score, voxel IoU,
  symmetric Chamfer and coverage are not yet computed.
- The reference is the ground-truth depth of the same steps, fused. It
  therefore cannot penalise a method for missing what no view could see, and it
  is not comparable to DAPS. A mesh-derived occupancy is required for the
  headline table.
- View subsets are evenly spaced with no seed, so no variance over subset
  choice is available.
