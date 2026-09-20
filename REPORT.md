# Report

Accumulated per stage: hypothesis, what was implemented, the result, how it is
read, what failed, and the next experiment. Numbers are means over the 39
held-out sequences of the three test scenes unless stated otherwise. Every
result file carries its config and the commit that produced it.

---

## Stage A (E00-E07): does reconstruction really get worse with more views?

**Hypothesis under test.** The earlier observation in this repository was that
fusing more posed views makes the reconstruction worse. That was measured with
accuracy alone, on evenly spaced view subsets, on a reference built from the
same steps. Before building anything on it, it has to survive a full metric set,
several scenes, and randomly chosen subsets.

**Implementation.** `src/stage_a.py`. One pass per sequence unprojects every
step once and then evaluates every (number of views, subset seed) combination on
that material. Metrics (`src/fuse.py:metrics`): accuracy (mean nearest distance
predicted to reference), completeness (reference to predicted), symmetric
Chamfer, precision / recall / F1 at 0.1, 0.2 and 0.5 m, voxel IoU on a shared
0.1 m grid, and the number of reconstructed points. Rankings: cross-view support
(point count per voxel) and the oracle (true distance to the reference), each at
kept fractions 1.0, 0.9, 0.75, 0.5, 0.25, 0.1, which is the
accuracy-completeness Pareto curve and the oracle headroom in the same pass.
Subsets: seed 0 is evenly spaced, seeds 1 and 2 draw the views at random.
Observation set r2 (front binaural pair), 0.1 m voxels, pixel stride 2.
8,892 rows in `results/E01_view_curve/r2/`.

**Result.** Fused cloud, all voxels:

| N views | accuracy (m) | completeness (m) | Chamfer (m) | F1@0.2 | IoU | points |
|---|---|---|---|---|---|---|
| 1 | 0.455 | 0.372 | 0.414 | 0.521 | 0.109 | 8.9k |
| 2 | 0.495 | 0.329 | 0.412 | 0.527 | 0.114 | 15.6k |
| 4 | 0.496 | 0.264 | **0.380** | **0.538** | **0.120** | 28.3k |
| 8 | 0.533 | 0.245 | 0.389 | 0.520 | 0.115 | 46.0k |
| 16 | 0.575 | 0.238 | 0.407 | 0.492 | 0.106 | 67.9k |
| 32 | 0.601 | 0.245 | 0.423 | 0.478 | 0.102 | 81.3k |
| all (23.7 mean) | 0.605 | 0.246 | 0.425 | 0.475 | 0.102 | 84.0k |

Support-ranked top quarter: accuracy 0.230 (N=1) to 0.287 (N=32), F1 peaking at
N=4 (0.594) and falling to 0.563. Oracle-ranked top quarter: accuracy 0.049 to
0.076, F1 rising monotonically 0.660, 0.715, 0.797, 0.841, 0.855, 0.854.

Seeds agree: F1@0.2 at N=4 is 0.549 / 0.547 / 0.518 for seeds 0 / 1 / 2, and the
ordering across N is identical for all three. All three scenes show the same
shape, with the turning point at N=2-4 (best F1 per scene: apartment_2 0.483 at
N=1, frl_apartment_5 0.537 at N=4, office_4 0.606 at N=2). Per
sequence-and-seed, 87 % have their best F1 at N of 4 or fewer.

**Reading.** The earlier claim was too strong and is corrected here. Accuracy
degrades monotonically with N, but completeness improves (0.372 to 0.238) and
saturates around N=8. The two together give a curve with an optimum: symmetric
Chamfer, F1 and IoU all peak at N=4 and decline after. So a few views help and
further views hurt, which is saturation-then-degradation rather than plain
degradation, and it is consistent across three scenes and three subset seeds,
so it is not an artefact of one trajectory or one metric.

The oracle column is the finding that matters. The oracle's F1 rises with every
added view, 0.660 at N=1 to 0.855 at N=16, and stays there. The material for a
better reconstruction therefore does keep accumulating as views are added. What
does not accumulate is any realisable ranking's ability to find it: the support
ranking peaks at N=4 and then falls back, so the whole benefit of views 5 and
beyond is present in the point set and lost in the selection.

**Failure to note.** The reference used here is the fused ground-truth depth of
the same steps, so it cannot penalise a method for missing geometry that no view
could see, and it is not comparable with published numbers. The headline table
needs a mesh-derived occupancy (E80-E82).

**Next.** Stage B: build the strongest point-only fusion that does not need a
posterior (outlier removal, clustering, trimmed consensus, keyframes, TSDF) and
see how much of the oracle gap it closes. If it closes most of it, the posterior
hypothesis is not needed and the direction changes (STOP CHECK A).

---

## Stage B (E10-E15): STOP CHECK A — how far does point-only robust fusion get?

**Hypothesis under test.** Before claiming a depth posterior is needed, the
strongest fusion that does not need one has to be built and measured. If it
closes most of the gap between counting support and the oracle, the posterior
hypothesis is not needed and the direction changes.

**Implementation.** `src/stage_b.py`. Ten families, each swept over its
parameter, all acting on the same fused voxel cloud so they differ only in which
voxels they keep: union; top fraction by support; at least k contributing
points; statistical outlier removal (mean distance to the 20 nearest voxels
above mean + z sd); radius outlier removal (fewer than n neighbours within r);
26-connected components of at least m voxels; a consensus filter on the spread
of the points inside each voxel; keyframes (views at least delta apart); a
truncated signed distance surface at several weight thresholds; and the oracle
ranking for reference. Robust operations are all in world geometry, never
across ERP pixels, which have no correspondence between views. 3,432 rows over
the 39 held-out sequences, at N = 4 and at every view.

**Result.** Best F1@0.2 per family, with the operating point that achieved it:

N = every view (23.7 mean):

| method | param | F1@0.2 | accuracy | completeness | Chamfer | IoU | kept |
|---|---|---|---|---|---|---|---|
| oracle | top 25 % | 0.854 | 0.076 | 0.275 | 0.175 | 0.308 | 0.25 |
| support, at least k | 4 | **0.566** | 0.317 | 0.433 | 0.375 | 0.145 | 0.34 |
| support, top fraction | 0.25 | 0.562 | 0.287 | 0.542 | 0.415 | 0.150 | 0.25 |
| radius outlier removal | 0.15 m / 16 | 0.538 | 0.310 | 0.557 | 0.434 | 0.106 | 0.18 |
| TSDF | weight 50 | 0.531 | 0.260 | 0.858 | 0.559 | 0.147 | 0.27 |
| keyframe | 1.0 m | 0.526 | 0.478 | 0.376 | 0.427 | 0.124 | 0.35 |
| statistical outlier removal | z = 0 | 0.517 | 0.426 | 0.365 | 0.395 | 0.121 | 0.66 |
| connected components | 500 | 0.479 | 0.587 | 0.254 | 0.420 | 0.103 | 0.98 |
| consensus (voxel spread) | 0.06 | 0.476 | 0.605 | 0.246 | 0.425 | 0.102 | 1.00 |
| union | – | 0.475 | 0.605 | 0.246 | 0.425 | 0.102 | 1.00 |

N = 4: oracle 0.814, best point-only support at least 3 → 0.621, union 0.549;
TSDF 0.600, radius outlier removal 0.600, statistical outlier removal 0.590,
connected components 0.559, keyframe 0.550, consensus 0.549.

Gap closed, (best point-only − union) / (oracle − union): **24 %** at every view
and **27 %** at N = 4.

**Reading.** STOP CHECK A does not fire. The best point-only fusion closes about
a quarter of the oracle gap and leaves three quarters, so the observation that
motivates this project is not explained away by weak fusion.

Two further findings, both worth stating plainly. First, none of the
conventional robust operations beats simply counting how many views put a point
in a voxel: statistical and radius outlier removal, connected components, the
voxel-spread consensus filter and keyframing all land at or below it, and only
the TSDF comes close at N = 4. What separates a real surface from a wrong one
here is how many independent views assert it, not whether it sits in a dense or
well-connected part of the cloud. Second, robust fusion does not remove the
view-count problem: the best point-only result at N = 4 (0.621) is still better
than the best at every view (0.566), so more views continue to hurt even under
the strongest selection available without a posterior.

**Failure to note.** The consensus filter as implemented is nearly inert (its
best setting keeps 100 % of voxels), because the spread of points inside a
0.1 m voxel is bounded by the voxel itself. A per-ray consensus across views,
rather than a per-voxel one, would be the meaningful version and is not
implemented.

**Next.** The fake Gaussian posterior control, which needs no training: turn the
existing point predictions into posteriors of fixed width and fuse them softly.
That separates "posterior fusion helps" from "learned ambiguity helps" before
any model is trained, and it exercises the posterior fusion code that Stage D
needs.

---

## Stage B addendum (E25): the fake Gaussian control, on matched candidates

**Why.** The first run of this control compared the best operating point of each
method, but the posterior scored a full room grid (up to 1.5 M voxels) while the
support ranking scored only voxels some view's argmax had hit (~84 k). Best-of
over different candidate sets is not a comparison.

**What was done.** `src/stage_b2.py` now also scores the *same* voxels the
support ranking sees (`fake_posterior_same_cand`), so soft evidence is compared
against hard counting with the candidate set held fixed. 6,240 rows.

**Result.** F1@0.2 at matched kept fractions, every view:

| kept fraction | support | fake posterior (sigma 0.05) | oracle |
|---|---|---|---|
| 0.002 | 0.053 | 0.064 | 0.209 |
| 0.010 | 0.143 | 0.148 | 0.537 |
| 0.050 | 0.383 | 0.331 | 0.781 |
| 0.100 | 0.487 | 0.446 | 0.827 |
| 0.250 | **0.562** | 0.540 | **0.854** |
| 0.500 | 0.549 | 0.537 | 0.726 |

Accuracy at the tight end favours the posterior (0.076 m against 0.256 m at
fraction 0.002) and completeness favours counting throughout.

**Reading.** On matched candidates and matched budget the two are the same
method to within 0.02 F1, crossing over around a kept fraction of 0.02: soft
evidence is the better precision ranking at the extreme tail and slightly worse
everywhere that matters. Neither recovers the oracle gap. So STOP CHECK B's
first half is settled before any training: **soft accumulation of a blurred
point buys nothing.** If a learned posterior helps later, the credit belongs to
the shape of the distribution, not to softness. The earlier "-55 %" figure is
withdrawn; its cause was found in a later audit (2026-09-20) and was not the
candidate-set mismatch this section first blamed: the run's summary averaged
the fake rows over all eight kept fractions before taking the best per method,
while the support rows were already per fraction. Summarised per operating
point (`results/E25_fake_posterior/r2/metrics.json`, regenerated), the fake
posterior at its own best fraction is 0.540 on matched candidates and 0.539 on
the room grid against support 0.562 at every view, and 0.590 / 0.570 against
0.611 at N=4: about -8 % to -20 % of the oracle gap, never positive. The table
above, which was computed per fraction from the start, is unchanged.

---

## Stage C (E20-E25): does the posterior keep the true depth alive?

**Hypothesis under test.** If single-view echo geometry is ambiguous, a
distribution trained on the same backbone should carry the true depth as a
secondary hypothesis on the rays where the point estimate is wrong. If it does
not, the loss is in the single-view predictor, not in the fusion.

**Implementation.** `src/posterior_head.py` replaces only the decoder's final
1-channel convolution with a K=128 linear-metric-bin head; the backbone,
attention stack and decoder are the base model's, loaded from its released
checkpoint. Trained by `src/train_posterior.py` on the base model's own training
split (12 scenes, scene-disjoint from the held-out ones), KL against a Gaussian
soft label one bin wide, two epochs head-only then full fine-tuning, 20 epochs,
no auxiliary expected-depth term. Evaluated by `src/eval_posterior_rays.py` on
the full test split (133.7 M rays) with temperature fitted on validation (1.462)
and never on test.

**Result.** Learned posterior, r2:

| quantity | value |
|---|---|
| argmax MAE | 0.246 m (the point baseline reports 0.289 m on the same split) |
| expected-depth MAE | 0.238 m |
| argmaxes wrong by > 0.2 m | 26.5 % |
| NLL / Brier / ECE | 2.400 / 0.829 / 0.160 |
| mode coverage @1 / @10 | 72.4 % / 74.9 % |
| **mode rescue @3 / @5 / @10** | **7.1 % / 8.1 % / 8.6 %** |
| modes per ray (above 10 % of the peak) | 1.28 |
| mass within 0.2 m of the truth, on wrong rays | 15.1 % |

Fake Gaussian on the same argmax, every sigma: mode rescue 0.0 % at every K,
1.00 modes per ray, NLL 3.33 to 7.68. Bin-based rescue@10 is 44.9 % for the
learned posterior and 40.9 % for the fake.

**Reading.** Three things, in order of how much they matter.

The bin-based Rescue@K in the plan cannot answer the question it was written
for. Ten bins of 0.077 m span ±5 bins around the argmax, so a *unimodal*
Gaussian scores 40.9 % on it. Ranking local maxima instead separates the two
cleanly, and every number below uses that version.

The learned posterior is genuinely multimodal where the fake cannot be: 1.28
modes per ray against 1.00, and mode rescue 8.6 % against 0.0 %. So the head did
learn something a blurred point does not contain, and STOP CHECK B does not fire
on the mode metric.

But the size is small, and this is the number that shapes the next stage. On the
26.5 % of rays where the point estimate is wrong, a secondary mode carries the
truth 8.6 % of the time, which is 2.3 % of all rays. Mode coverage rises only
from 72.4 % at K=1 to 74.9 % at K=10. Read strictly, this is close to the plan's
CASE 2: most of what the point estimate loses is not recoverable from the
distribution either, so the ceiling on posterior fusion is set by the
single-view predictor.

One qualification against that strict reading: fusion does not need a discrete
second mode, it needs probability mass in the right place, and on wrong rays the
posterior still puts 15.1 % of its mass within 0.2 m of the truth against the
fake's 43.8 % at sigma 0.2 (which is mass around a *wrong* centre). Whether
15.1 % of correctly-placed mass, accumulated over views, is enough to beat
counting is exactly what Stage D measures, and it is no longer a matter of
opinion.

**Failure to note.** Validation KL bottoms at epoch 2 (1.346) and rises
monotonically afterwards while the argmax error keeps improving slightly
(0.292 to 0.274), so the checkpoint chosen by KL is an early one. With 12
training scenes the distribution overfits well before the point estimate does.
Choosing on KL is the honest choice for a distributional model and is kept, but
the trade is recorded here.

**Next.** Stage D, the comparison the project exists for, on one backbone:
point regression to fusion, distribution to argmax to point fusion, and full
posterior fusion, over the view-count curve.

---

## Stage D (E30-E32, E40-E42): point, argmax and full posterior on one backbone

**Hypothesis under test.** The project's central claim. If collapsing an
ambiguous echo into a point depth is what makes extra views harmful, then
keeping the distribution and aligning it in world space should turn the
view-count curve from falling into rising, while taking the argmax of the same
distribution should reproduce the falling curve.

**Implementation.** `src/stage_d.py`, one backbone, three readouts, the same
sequences, the same metric set, the same kept-fraction sweep.

- **A point** — the base model's point regression, unprojected, voxels ranked by
  cross-view support.
- **B argmax** — the posterior head's argmax depth through exactly the same
  pipeline, so A and B differ only in which network produced the depth.
- **C posterior** — the full distribution, soft surface consensus
  (`O_i(x)` summed over views, no free-space carving), ranked by accumulated
  evidence. Scored on two candidate sets: `C_on_B`, exactly the voxels B
  proposes, which isolates soft evidence from hard counting, and `C_grid`, a
  voxel grid over the room, which lets the posterior support geometry no argmax
  proposed.

The posterior's bins are per-face cubemap depth, so the radial query band is
scaled by each ray's face factor; a unit test shows that skipping this loses the
most oblique ray entirely (factor 0.596). 7,020 rows, 39 sequences.

**Result.** Best F1@0.2 over the kept-fraction sweep:

| method | N=1 | 2 | 4 | 8 | 16 | all |
|---|---|---|---|---|---|---|
| A point | 0.572 | 0.600 | **0.611** | 0.592 | 0.574 | 0.562 |
| B argmax | 0.576 | 0.600 | **0.618** | 0.596 | 0.570 | 0.556 |
| C posterior, on B's voxels | 0.563 | 0.596 | 0.608 | 0.573 | 0.543 | 0.526 |
| C posterior, room grid | **0.595** | **0.606** | 0.600 | 0.568 | 0.538 | 0.524 |
| oracle | 0.719 | 0.770 | 0.834 | 0.859 | 0.867 | 0.861 |

At N=4 the operating points are: A F1 0.611 (accuracy 0.247, completeness
0.513), B 0.618 (0.253 / 0.452), C_on_B 0.608 (0.317 / 0.398), C_grid 0.600
(0.359 / 0.358), oracle 0.834 (0.055 / 0.265).

**Reading. The central hypothesis is not supported.**

E31 is satisfied, which makes the comparison clean: at N=1 the point model and
the posterior's argmax are the same predictor to within 0.004 F1 (0.572 against
0.576), so any difference at higher N belongs to the representation and not to
one network being better.

Keeping the distribution does not change the shape of the curve. C peaks at
N=2-4 and falls exactly as A and B do, and from N=4 on it is *worse* than the
argmax of the same distribution (0.526 against 0.556 at every view). The
predicted pattern — point falling, argmax falling, full posterior rising — does
not appear in any column. This is the plan's CASE 4, and in a stronger form than
CASE 4 describes: preserving the posterior is not merely equivalent to
collapsing it, it costs something once several views are accumulated.

There is one real but small effect in the predicted direction. With a single
view, posterior fusion over the room grid is the best of the four (0.595 against
0.572), because it can place surface where no argmax landed. That advantage is
gone by N=4 and reversed after.

Why it fails is consistent with Stage C. Only 8.6 % of wrong rays carry a
correct second mode, so there is little right-place mass for views to reinforce,
while the bulk of each posterior's mass sits around its own argmax — which is
wrong on 26.5 % of rays. Summing that mass over many views accumulates
wrong-place evidence smoothly, whereas counting points at least demands a hard
coincidence in one 10 cm voxel before it credits anything. Softness helps when
the truth is inside the spread and hurts when it is not, and here it is not,
often enough to lose.

**What still stands.** The oracle keeps rising with every view, 0.719 to 0.867,
so the material for a better reconstruction does accumulate. Nothing tried so
far — robust point fusion (24 % of the gap), a fake Gaussian posterior (0 %), a
learned posterior (negative) — converts it into a usable ranking.

**Failure to note.** C's parameters are not tuned. tau is fixed at 0.15 m, bins
at 128, no truncation study, weights uniform, and the room grid is coarsened
when a sequence's bounding box is large. E42 and E43 are therefore not done, and
it is possible, though not indicated by anything measured, that a different tau
or a finer grid changes the ordering. What is not in doubt is the *shape*: no
setting of tau changes that C falls with N in the same way A and B do.

**Next.** This is a decision point for the owner, not one the agent should take.
Three directions remain open and they are mutually exclusive in effort:
Stage E (occupied / free / unknown fusion, where the acoustic-specific
lambda_free sweep might change what evidence means), Stage G (a learned
evidence weighting, which the plan puts after a working deterministic fusion —
and there is now no working deterministic fusion to put it after), or accepting
that the single-view predictor is the binding constraint and improving it
instead. The result above says the paper's current claim cannot be made.

### Stage D on the eight-observation model (posterior_r8)

Same script, same 39 sequences, the `r8` point predictions and the `posterior_r8`
head (best epoch 2, val KL 1.190; temperature fitted on val 1.343, not applied
here — see the note below). Ray level first (E21, 133.7 M test rays): argmax
MAE 0.209 m against 0.246 m for r2, 23.7 % of argmaxes wrong by more than 0.2 m,
mode rescue@10 7.1 %, 1.19 modes per ray, mass at the truth when the argmax is
wrong 16.3 %, fake Gaussian rescue 0 % at every sigma. More microphones sharpen
the argmax and do not add a second mode.

| method | N=1 | 2 | 4 | 8 | 16 | all |
|---|---|---|---|---|---|---|
| A point | 0.614 | 0.642 | **0.651** | 0.631 | 0.603 | 0.590 |
| B argmax | 0.604 | 0.630 | 0.636 | 0.614 | 0.588 | 0.574 |
| C posterior, on B's voxels | 0.579 | 0.615 | 0.627 | 0.599 | 0.569 | 0.554 |
| C posterior, room grid | 0.601 | 0.629 | 0.622 | 0.589 | 0.560 | 0.546 |
| oracle | 0.745 | 0.792 | 0.837 | 0.855 | 0.855 | 0.848 |

The ordering is the same as for r2 at every view count: point ≥ argmax >
posterior, all three peaking at N=4 and falling after, the oracle rising to
0.855. Two differences from r2. The single-view advantage of grid posterior
fusion is gone (0.601 against 0.614; for r2 it was 0.595 against 0.572). And
the posterior's argmax is now slightly *worse* than the point model at N=1
(0.010 F1), so E31 holds less exactly than for r2; the B-versus-C comparison,
which is within one network, is unaffected. With both observation sets the
conclusion of the section above stands.

**Two things Stage D did not do, to be corrected before any further use.**
The band mass was computed from the raw softmax (temperature 1), whereas Stage
C's calibration numbers used the val-fitted temperature (1.462 for r2, 1.343 for
r8). A temperature above 1 broadens every ray's distribution, so the fused
evidence would be smoother still; the direction of the effect on the ranking is
not obvious and has to be measured. And the room grid's bounding box was taken
from the reference and B's voxels together; the reference is ground truth and
should not set even the box. Both are fixed in the next run.

---

## Stage D, second run (v2): after the audit fixes, plus per-view confidence before fusion

**Why.** A read-only audit (2026-09-20) of Stages A-D found three things that all
leaned against the posterior: band mass counted only bins wholly inside the
band (at K=128 and tau=0.15 the band shrank to about 2.3 of its 3.9 nominal
bins in face space, to one bin or less on 6 % of rays), the softmax temperature
was fitted on 48 frames of one validation scene and then not applied in Stage
D at all, and the room-grid box borrowed its extent from the reference. It also
found that the kept fraction was chosen on the test sequences, that the grid's
optimum at N=1 sat on the edge of the sweep, and that the Stage B2 "-55 %"
line came from a summary bug (corrected above). The owner separately asked
whether a confidence for each 2-D depth map should be estimated *before* any
3-D fusion. All of this is answered by one re-run.

**What changed.** `posterior.ViewPosterior` interpolates the cdf at the band
ends (unit tests: sub-bin band keeps proportional mass; random poses
round-trip to 0.003). The temperature is refitted on 1,000 rays from every
validation frame (r2 1.378, r8 1.302; previously 1.462 / 1.343) and applied.
The grid box comes from the predictions only, and the grid sweep goes down to
0.005. Validation-scene sequences were predicted (37 sequences) and every
method's kept fraction is now chosen there (`src/select_frac.py`); the test
number at that fraction is reported. Two new rankings use the per-ray
confidence c = max softmax probability after temperature: **D** ranks B's
voxels by the *sum of c* over the rays that hit them instead of the ray count,
and **E** drops each view's rays below its 50th / 75th confidence percentile
before counting. Ray-level confidence-versus-error is recorded on 18.5 M test
rays per model. Runs: `results/E30_stage_d/{r2,r8}_v2` (test),
`{r2,r8}_v2_val` (validation), 10,800 rows each, one deterministic evenly
spaced view subset per N as before.

**Ray level: confidence is a good error predictor.** Spearman between
confidence and argmax error is -0.66 (r2) / -0.65 (r8). The fraction of rays
wrong by more than 0.2 m falls from 84 % in the lowest confidence decile to
0.8 % in the highest (r2; r8 83 % to 0.6 %). Keeping the most confident half of
the rays cuts MAE from 0.321 m to 0.081 m, against 0.033 m for an oracle
ordering by true error (AUSE 0.054 m; r8 0.277 → 0.072, AUSE 0.048). On the
validation scenes the same holds (Spearman -0.61 / -0.59).

**Fusion level: test F1@0.2 at the validation-selected kept fraction.**
(The test-side maximum differs from these by at most 0.015, so the earlier
test-chosen numbers were not materially optimistic.)

r2:

| method | N=1 | 2 | 4 | 8 | 16 | all |
|---|---|---|---|---|---|---|
| A point | 0.572 | 0.600 | 0.610 | 0.592 | 0.574 | 0.562 |
| B argmax | 0.576 | 0.600 | 0.609 | 0.596 | 0.570 | 0.556 |
| C posterior, on B's voxels | 0.574 | 0.602 | 0.605 | 0.570 | 0.537 | 0.520 |
| C posterior, room grid | 0.591 | 0.612 | 0.597 | 0.559 | 0.526 | 0.512 |
| D confidence-weighted support | 0.588 | 0.613 | 0.617 | 0.590 | 0.557 | 0.549 |
| E drop least-confident 50 % of rays | 0.424 | 0.449 | 0.473 | 0.443 | 0.412 | 0.400 |
| E drop least-confident 75 % | 0.268 | 0.285 | 0.320 | 0.293 | 0.268 | 0.258 |
| oracle | 0.719 | 0.770 | 0.834 | 0.859 | 0.867 | 0.861 |

r8:

| method | N=1 | 2 | 4 | 8 | 16 | all |
|---|---|---|---|---|---|---|
| A point | 0.614 | 0.642 | 0.651 | 0.631 | 0.603 | 0.590 |
| B argmax | 0.604 | 0.630 | 0.636 | 0.614 | 0.586 | 0.574 |
| C posterior, on B's voxels | 0.589 | 0.619 | 0.621 | 0.592 | 0.564 | 0.549 |
| C posterior, room grid | 0.614 | 0.633 | 0.614 | 0.577 | 0.547 | 0.532 |
| D confidence-weighted support | 0.613 | 0.638 | 0.642 | 0.617 | 0.589 | 0.573 |
| E drop least-confident 50 % | 0.435 | 0.463 | 0.485 | 0.454 | 0.426 | 0.412 |
| E drop least-confident 75 % | 0.277 | 0.299 | 0.328 | 0.303 | 0.280 | 0.271 |
| oracle | 0.745 | 0.777 | 0.837 | 0.855 | 0.855 | 0.848 |

**Reading.**

1. *The three fixes did not change the Stage D conclusion.* With the band
   interpolated, the temperature applied and the box clean, the full posterior
   still peaks at N=2-4 and falls, and from N=4 on is below the argmax of the
   same distribution by 0.01-0.04 (r2 all views 0.520 against 0.556; r8 0.549
   against 0.574). The audit's worry that the bugs had piled up against C is
   settled: they had not. CASE 4 stands.
2. *Per-ray confidence is real but does not move the curve.* Weighting the
   support count by confidence (D) is within ±0.01 of plain counting at every
   N, peaks at N=4 like everything else, and falls after. It is the best
   non-oracle ranking at N=1-4 for r2 by 0.006-0.016, which is inside the
   spread between view subsets seen in Stage A (0.03).
3. *Filtering by confidence is much worse, and the reason is the finding.* At
   N=4 (r2) dropping the least-confident half of each view's rays raises
   precision@0.2 from 0.685 to 0.770 but cuts recall from 0.556 to 0.345 and
   completeness from 0.51 m to 0.97 m, because the low-confidence rays are the
   far and oblique surfaces and nothing else covers them. The confident half of
   the rays is accurate (MAE 0.08 m) yet only 77 % of its *voxels* are within
   0.2 m of the reference: correct rays pile into shared voxels while the few
   wrong ones scatter into voxels of their own, so a 3 % ray error rate
   becomes a 23 % voxel error rate. That amplification is what the support
   count already corrects, which is why nothing built on confidence can beat
   it by much.
4. *What the answer to the owner's question is.* Estimating uncertainty per
   2-D depth map first is worth doing for what it tells you — it separates
   wrong rays well, cheaply, with a head that already exists — but not as a
   pre-filter for 3-D fusion on this data: the rays it would remove are the
   only source of the surfaces that make up half the reference. The binding
   constraint is the single-view predictor on far and oblique rays, not the
   fusion rule, and no reweighting of the same rays recovers what those rays
   do not carry. The oracle's rise with N shows the missing surfaces are
   *present* in the union of predictions at some views; the problem is that
   each such view also brings wrong surfaces at the same rate, and no
   per-ray or per-voxel signal tried so far tells them apart.

**Disclosures that were missing.** The reference for each N is the fused ground
truth of the same N views, so each column is measured against its own
reference; "F1 falls with N" means the prediction worsens relative to what
those views could have reconstructed. All Stage B-D numbers use one
deterministic view subset per N (seed 0, evenly spaced); Stage A's three
seeds put the subset-to-subset spread at about 0.03 F1, comparable to the
differences between the non-oracle methods here. The room grid is coarsened
to at most 0.121 m on 32 % (r2) / 14 % (r8) of C_grid rows, which makes its
IoU and 0.1 m metrics not strictly comparable to the point-cloud methods;
F1@0.2 ordering is unaffected. The Stage D oracle is computed on B's voxels
and is the ceiling for A/B/C_on_B/D/E, not for C_grid. The posterior head's
"initialisation from the point head" is a uniform start, not a warm start
(`src/posterior_head.py`). The earlier Stage D tables (`results/E30_stage_d/{r2,r8}`)
are the T=1, whole-bin version and are kept for the record.
