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
