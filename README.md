# EchoRecon

Audio-driven 3D reconstruction from a *sequence* of posed echo observations.

The base model (a panoramic echo-depth network kept in a local sibling
checkout, referred to here only as *the base model*; its path is set by
`ECHO_DEPTH_ROOT`) predicts a listener-centred equirectangular (ERP) depth map
from the echoes heard at one position. EchoRecon asks what a sequence of those
predictions gives: the receiver moves along a trajectory with known poses, every
step yields an ERP depth with a certainty, and the depths are unprojected into
one world frame and fused into a 3D model, weighting each observation by how
much it can be trusted where several of them see the same surface.

The base model's code is imported from the sibling checkout at run time, never
copied here. Which of its commits this work pins to is **undecided** until its
remote state has been checked; the sibling is used as it is on disk for now.

## Plan

1. **Geometry sanity with ground truth.** Unproject the rendered ERP depths of
   a sequence at their poses and check that they agree with each other, so the
   ERP convention, the pose quaternion and the radial depth line up
   (`src/convention_check.py`, `src/map_check.py`).
2. **Predicted depth, naive fusion.** Run the base model on every step of a
   sequence, unproject, accumulate into a point cloud and a voxel model
   (`src/predict.py`, `src/eval_fusion.py`). Compare with the ground-truth fusion
   of the same sequence: accuracy, completeness, per-step depth error before and
   after fusion.
3. **Certainty-weighted fusion.** Per-step, per-ray reliability from (a) the
   model's own signal (observation-dropout variance or an uncertainty head),
   (b) multi-view agreement across steps, (c) range and grazing angle; fuse with
   those weights and measure what each source buys over the baselines below.

## Baselines that every fusion result is reported against

- **Uniform**: every ray of every step weighted 1 (naive averaging).
- **Range-prior only**: weight is a fixed decreasing function of the predicted
  range and nothing else. This is the trivial baseline; if a learned or
  model-derived confidence does no better than this, the confidence has
  reduced to a distance prior and is not carrying information.

## Prior predictions and evaluation criteria

Recorded before the experiments, so the outcome can be judged against them
rather than against thresholds chosen afterwards.

- Prior (the project owner, 2026-09-18): fusing a sequence of predictions with
  known poses should raise accuracy over the single-step prediction
  substantially ("굉장히 오르지 않을까").
- Success criteria: **undecided** (기준 미정). The placeholders below are to be
  filled in by the owner, not by the agent running the experiments:
  - fused accuracy improves over single-step by at least X
  - certainty-weighted fusion improves over the range-prior baseline by at least Y

## Hypotheses under verification (not established)

- Reliability is anisotropic around the receiver in a way that is not a
  function of range alone (a "shell" whose thickness depends on direction).
  Status: hypothesis, to be tested in step 3; not to be assumed by any code path.

## Sweepable from the start

- STFT hop of the base model's input (`--hop`, default the base model's trained
  value); every prediction file records the hop it was made with.
- ERP convention, fusion voxel size, truncation, weighting function: all
  arguments, all recorded in the output's metadata.

## Data

`$FORWARD_SEQ_ROOT = /root/storage/replica_0422_forward_seq/<scene>/seq_XXXX/`:
`poses.json` (habitat position xyz with y up; rotation quaternion w,x,y,z about
y; heading in degrees; 0.15 m steps), `audio_wav/step_XXX_rel{000,090,180,270}.wav`
(binaural echo at four receiver headings, 48 kHz), `erp_depth_radial/step_XXX.npy`
(512x1024 float32 metres, radial), `erp_rgb/step_XXX.png`. 18 Replica scenes,
11-15 sequences each, 208-397 steps per scene. Held-out scenes follow the base
model's split: val {apartment_1, frl_apartment_4, office_3}, test {apartment_2,
frl_apartment_5, office_4}.

Out of scope for now: sensitivity to pose error (poses are taken as known) and
simulator limits.

## Layout

```
src/data.py              sequence loader (poses, GT ERP depth, audio)
src/erp.py               ERP ray directions, unprojection, habitat pose -> world
src/fuse.py              point-cloud accumulation, voxel occupancy, weighted TSDF, metrics
src/convention_check.py  step 1: consecutive-frame agreement per ERP convention
src/map_check.py         step 1: fused ground truth against the scene's floor plan
src/predict.py           base-model predictions for every step of a sequence -> npz (hop recorded)
src/eval_fusion.py       steps 2/3
outputs/                 predictions, fused models, reports (not versioned)
```

Environment: `/opt/conda/envs/shared_audio` (torch 2.8, scipy, trimesh, soundfile).

## Step-1 result so far

apartment_2/seq_0000, ground-truth depths of 13 consecutive steps, stride-4
pixels: with the ERP centre column as the forward axis the nearest-neighbour
distance between consecutive unprojected frames has median 1.5 cm and 89.6 % of
points within 5 cm; with the centre column as the backward axis it is 1.9 cm /
70.1 %. The azimuth sign is not decidable from frame agreement alone (a mirrored
world is equally self-consistent); against the scene's floor plan, wall-height
points of the fused ground truth (20 steps) land within 10 cm of a plan wall
for 94.8 % of points under *centre column = forward, azimuth increasing to the
right* ("right0") and for 25-37 % under the other three. So the ground-truth
depths of a sequence do fuse into one consistent 3D model, and "right0" is the
default convention from here on.

## Step 1: the oracle ceiling and what the support ranking recovers

**Why.** Fusing every predicted point is worse than a single step, and keeping
the voxels with the most cross-view support is better. That raises the question
the ceiling answers: is the remaining error there because the information is not
in the predictions, or because the ranking is poor? An oracle ranking (sort the
fused voxels by their true distance to the reference) separates the two.

**What was done.** For each of the 39 held-out sequences, all steps are fused
into one voxel set whose positions are the plain average of the points in each
voxel. Four rankings are then scored on those same voxels: cross-view support
(point count), a range prior, the observation-dropout spread, and the oracle.
Each ranking's top-fraction curve is measured, and the recovery rate at fraction
f is (uniform − method) / (uniform − oracle). Observation sets r2 / fb / r6 / r8
(one, two, three, four receiver headings), trained hop 160, 0.1 m voxels,
tau = 0.2 m, 23.7 steps per sequence on average.

**Result.** Accuracy as mean nearest distance to the reference and the fraction
within 0.2 m; completeness reported separately.

| observation set | ERP MAE | single step | fused, all | support, top 25 % | oracle, top 25 % |
|---|---|---|---|---|---|
| r2, one heading | 0.296 | 0.44 / 51 % | 0.60 / 35 % | 0.29 / 60 % | 0.08 / 97 % |
| fb, two headings | 0.286 | 0.44 / 52 % | 0.62 / 36 % | 0.27 / 61 % | 0.07 / 97 % |
| r6, three headings | 0.279 | 0.43 / 52 % | 0.59 / 37 % | 0.28 / 62 % | 0.07 / 97 % |
| r8, four headings | 0.253 | 0.40 / 55 % | 0.56 / 39 % | 0.26 / 65 % | 0.07 / 98 % |

Completeness (metres, reference → predicted), r2: all voxels 0.25, support top
25 % 0.54, oracle top 25 % 0.27.

Recovery rate, (uniform − method) / (uniform − oracle):

| ranking | top 75 % | top 50 % | top 25 % | top 10 % |
|---|---|---|---|---|
| range prior (r2 / r8) | +0.19 / +0.18 | +0.12 / +0.12 | +0.03 / +0.04 | −0.01 / +0.00 |
| dropout spread (r2 / r8) | −0.08 / +0.04 | −0.10 / +0.09 | −0.13 / +0.06 | −0.15 / +0.03 |

**Reading.** The information is in the fused set: a ranking that knows the
answer keeps a quarter of the voxels at 0.07 m with 97 % inside 0.2 m, and it
does so without paying in completeness (0.25 → 0.27 m, where the support
ranking pays 0.25 → 0.54 m). So the limit measured here is the ranking, not the
predictions. Cross-view support closes part of the gap on its own, and the two
cheap scores on top of it close almost none of the rest: the range prior
recovers a fifth of the gap at the loose end and nothing at the tight end, and
the dropout spread is not usable at all for r2 (where only one of two
observations can be dropped, so the spread is a weak signal, as expected) and
recovers under a tenth for r8. Adding receiver headings improves every column a
little and changes no conclusion: one heading to four moves the per-step ERP
error 0.296 → 0.253 m and the support-ranked quarter 0.29 → 0.26 m, while the
oracle stays at 0.07 m throughout.

**Against the number of views.** The same sequences fused from N steps spread
evenly over the trajectory (accuracy / fraction within 0.2 m, completeness in
metres):

| N | r2 fused all | r2 support top 25 % | r2 oracle top 25 % | r2 completeness |
|---|---|---|---|---|
| 2 | 0.48 / 47 % | 0.23 / 71 % | 0.05 / 100 % | 0.27 |
| 4 | 0.50 / 44 % | 0.25 / 69 % | 0.05 / 100 % | 0.24 |
| 8 | 0.53 / 40 % | 0.26 / 65 % | 0.06 / 100 % | 0.24 |
| 16 | 0.57 / 36 % | 0.28 / 61 % | 0.07 / 98 % | 0.24 |
| all (23.7) | 0.60 / 35 % | 0.29 / 60 % | 0.08 / 97 % | 0.25 |

r8 behaves the same way one step better throughout (N = 2: 0.44 / 51 %, all:
0.56 / 39 %). Note that "top 25 %" is a fraction, not a fixed budget, so the kept
set grows with N; the "fused all" and completeness columns are budget-free and
carry the finding.

**Reading.** Coverage saturates at two views: completeness is 0.27 m at N = 2 and
0.24-0.25 m from N = 4 on, so a sequence adds almost no new visible surface.
Accuracy meanwhile degrades monotonically with N, 0.48 -> 0.60 m. Each further
view therefore contributes mostly wrong surface on top of ground already
covered, which is what a prediction that spans the whole panorama at roughly the
right scale would do. Whatever a trajectory is worth here, it is not worth
accumulation; it has to be worth *disagreement*, i.e. using the later views to
reject what the earlier ones got wrong. That is what steps 3 and 4 test.
