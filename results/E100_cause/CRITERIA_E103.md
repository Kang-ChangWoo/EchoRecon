# Pre-registered criteria — E103 per-ray confidence under the fixed reference

Written 2026-09-21 before any E103 number was computed. E100–E102 exhausted the
geometric statistics of the fused cloud (support, free-space contradiction,
view count, baseline): none separates wrong voxels beyond point counting,
while the oracle ranking on the same voxels reaches F1 0.86. Stage D showed the
posterior head's max-softmax confidence ranks rays well (wrong rate 84.5 % →
0.8 % across deciles, Spearman −0.66) but that confidence-weighted fusion gained
≤ 0.01 and confidence filtering collapsed F1 — **both under the moving
reference and at the all-voxel operating point**. E103 re-scores exactly those
two uses of the confidence under the fixed reference and at the top-fraction
operating point.

## Method
Posterior heads `posterior_r2`, `posterior_r8` (best by val KL, temperatures
1.378 / 1.302 from `results/E21_posterior_rays/`), one forward pass per step:
argmax depth (the B pipeline of Stage D) and per-ray confidence c = max softmax.
On B's voxels (0.1 m, positions = plain mean), rankings
- `support`   point count (control),
- `views`     distinct view count (E102's best),
- `conf_sum`  Σ c over the points in the voxel,
- `conf_mean` mean c over the points (ties by count),
and one filter, `conf_filter50`: rays below the per-view median confidence are
dropped before voxelising (Stage D's E50), reported beside `support` at the
same kept count (`support_matchE50`). Fractions {1.0, 0.5, 0.25, 0.1}; headline
0.25; fraction chosen on val for the verdict row. Fixed reference; subset
reference recorded. N ∈ {1, 2, 4, 8, 16, all}, subset seed 0 (seeds 1–2 for
the base spread). Test scenes give every number; val chooses the fraction only.

## Verdict
- gain = F1@0.2(conf ranking) − F1@0.2(support), same voxels, same fraction,
  same N, paired over the 39 test sequences, fixed reference.
- *meaningful* if gain(all) ≥ 0.03 with the bootstrap 95 % CI above 0 on both
  r2 and r8; *partial* one set; *no effect* |gain| < 0.03.
- The filter is judged against `support_matchE50` only (same count).
- Also reported: precision and recall separately, the N curve of the best
  confidence ranking, and gain at N = 1 (a per-ray confidence can act where
  support cannot).
- If no effect: the per-ray confidence, like the geometry, does not carry the
  information the oracle uses, and the saturation at 0.56 / 0.59 is a property
  of the per-view prediction that neither fusion rule nor confidence fixes.
