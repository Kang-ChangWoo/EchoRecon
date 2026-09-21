# Pre-registered criteria — issue B (E101 confound, E102b distance-based view selection) and issue C (confidence paradox)

Written 2026-09-22 before any of these numbers was computed.

## Issue B1 — E101 re-analysis with k controlled
The owner recomputed `per_view.csv`: Spearman(distance to nearest included
view, new-voxel precision) = +0.40 overall but ≈ +0.10 at fixed k, and k ↔
distance r = −0.82 in the spread order. Added to `marginal_view.py`: within
each k (k ≥ 2) the Spearman over rows of that k, for each order; the pooled
within-k Spearman (weighted by rows); and the same for the random order, where
distance varies at fixed k by construction. **Reading rule**: H1 stands only if
the pooled within-k Spearman ≥ +0.3 in the random order; otherwise the E101
conclusion is replaced by "late views are worse, not near views".

## Issue B2 — E102b, distance-based view selection (the direct experiment)
From all T views of a sequence, adopt greedily only views ≥ Δ from every
adopted view (Δ ∈ {0.3, 0.5, 1.0} m; the first view is step 0). Controls at
the *same count n(Δ)*: a random subset (seed 0) and the n consecutive steps
centred in the sequence. Fixed reference, A_point pipeline, frac 1.0 (every
voxel, where the fall exists) and frac 0.25 (support top quarter), F1@0.2,
both r2 and r8, 39 test sequences. **Verdict**: the distance rule "restores
the curve" if F1(Δ-spaced) − F1(random, same n) ≥ 0.03 with CI above 0 on both
sets at frac 1.0; if < 0.03, distance selection is no better than any subset
of the same size, and the fall is a function of count, not spacing. Also
reported: F1(Δ-spaced) vs F1(all views) — which is expected to be positive at
frac 1.0 simply because fewer views are used, and is not the verdict.

## Issue C — the confidence paradox, measured (E107)
Posterior-head predictions (`<mode>_post`, argmax depth + confidence c). At
N = all, per sequence: the B cloud's voxels (0.1 m). A voxel is *lost* by the
per-view median-confidence filter (Stage D's E50) if none of its points
survives; *kept* otherwise. For every voxel: correct = within 0.2 m of the
fixed reference; range = mean distance to the contributing views; incidence =
angle between the mean viewing direction and the reference surface normal at
the nearest reference point (normal from a 16-neighbour PCA of the reference
cloud). Bins: range {<2, 2–4, ≥4 m}, incidence {<30°, 30–60°, ≥60°}.
**Measured claims**: (i) share of the lost *correct* voxels that are far
(≥ 4 m) or oblique (≥ 60°); (ii) per bin, the fraction of correct voxels that
are lost (the "sole source" rate); (iii) recall@0.2 of the kept cloud vs the
full cloud per range bin. **Reading rule**: the repository's explanation
("low-confidence rays are the only source of far and oblique surfaces") is
*supported* if (i) ≥ 0.6 and the lost fraction in the far or oblique bins is
≥ 2× that in the near-and-frontal bin; *not supported* otherwise, in which
case the collapse is a plain loss of coverage independent of geometry. Both
r2 and r8; per scene.
