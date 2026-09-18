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

## Step-2 result so far (one sequence, preliminary)

apartment_2/seq_0000, 37 steps, base model at its trained hop (160), 256x512
predictions, 0.1 m voxels, tau = 0.2 m, reference = the fused ground truth of the
same steps.

| model | acc (mean NN, m) | acc < 0.2 m | completeness (m) |
|---|---|---|---|
| single step, voxelised, mean over steps | 0.565 | 26 % | 1.745 |
| fused, all voxels (uniform = range-prior = model-std) | 0.840 | 19 % | 0.579 |
| fused, top 50 % voxels by weight sum: uniform / range-prior / model-std | 0.46 / 0.42 / 0.43 | 28 / 29 / 29 % | 0.65 / 0.67 / 0.70 |
| fused, top 25 %: uniform / range-prior / model-std | 0.31 / 0.30 / 0.29 | 34 / 35 / 36 % | 0.82 / 0.84 / 0.87 |

Per-step ERP MAE against the ground truth is 0.56 m on this sequence.

Reading: fusing every predicted point makes the model *less* accurate than a
single step (outliers from every step survive while the agreeing points merge),
so the useful quantity is the support a voxel gathers across steps: keeping the
best-supported quarter roughly doubles the accuracy fraction. The three weight
rules rank voxels almost identically at every kept fraction, so neither the
range prior nor the observation-dropout spread adds information beyond the
multi-view count on this sequence. The top-down picture
(`outputs/fusion/apartment_2/seq_0000.png`) shows why: each step's prediction
is a smooth shell around the receiver, and the shells stack into a blob along
the trajectory instead of the room's walls. This is one sequence; the numbers
are not a result yet. The criteria X and Y above are still to be set by the owner.
