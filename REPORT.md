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
