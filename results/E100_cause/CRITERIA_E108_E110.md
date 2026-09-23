# Pre-registered criteria — E108 expected-count normalisation, E109 continuity-aware fusion, E110 top-k per ray

Written 2026-09-23 before any of the three was computed. Owner's premise: the
observations are not i.i.d. multi-view but consecutive observations along a
path; the goal is to *raise* reconstruction F1, and the question is whether
the headline "more views do not help" breaks when the continuity is used.

Common protocol: released point predictions for the A pipeline (E108, E109),
flat posterior heads for E110; 0.1 m voxels, stride 2; r2 and r8; both
references (fixed = GT of every step, subset = GT of the selected views);
N ∈ {1, 2, 4, 8, 16, all}, subset seed 0; 39 test sequences; every
hyper-parameter (thresholds, fractions, k) chosen on the 36 val sequences by
F1@0.2 at N = all under the fixed reference; test read once. **Baseline** =
point fusion with the raw count, at the same operating point (same kept
fraction, or the same kept count for thresholded variants). Always reported:
Δ F1@0.2 vs baseline with the paired bootstrap 95 % CI (39 sequences), the
N-curve, and recall@0.2 of the reference in range bins {<2, 2–4, ≥4 m} and
incidence bins {<30, 30–60, ≥60°} (E107's definitions), both references.
Band 0.03; *meaningful* = Δ ≥ 0.03 with CI above 0 on both r2 and r8.

## E108 — expected-observation-count normalisation
Vote v(c) = number of rays in voxel c. Expected rays
E(c) = Σ_j vis(c, j) · A / (r_j² · Δω(φ_j)), A = (0.1 m)² (voxel cross-section,
normal unknown), Δω(φ) = (2π²/(H·W)) cos φ the ERP pixel solid angle at the
voxel's elevation φ in view j, r_j its range. Two visibility definitions:
(A) *predicted-surface*: vis = 1 if the voxel lies no further than view j's
predicted depth along its direction + 0.2 m (deployable); (A′) *GT-surface*
(diagnostic upper bound only, not a method). Score_geo = v / max(E, E_min);
E_min chosen on val from {0.5, 1, 2}. (B) *smoothing approximation*: E_smooth =
Gaussian-smoothed count of predicted points in 3-D (σ ∈ {0.3, 0.5, 1.0} m, val),
Score_smooth = v / E_smooth. Read as a ranking at frac 0.25 (vs raw count) and
as a threshold (rate ≥ t, t on val) vs raw count at the same kept count.
**Verdict**: meaningful if Δ ≥ 0.03 on both sets; the far-bin recall (≥ 4 m)
is reported as the targeted quantity whatever the F1 says.

## E109 — continuity-aware accumulation (three variants)
ρ(d) fixed from E100 (c): correlation of neighbouring views' errors 0.88 at
d < 0.5 m, 0.73 at 0.5–1.5 m, 0.41 at ≥ 1.5 m (piecewise, interpolated
linearly in d between bin centres, ρ(0) = 0.95, ρ(≥ 3 m) = 0.3).
a) *baseline-weighted*: views processed in trajectory order; a view's vote at
   a voxel is weighted (1 − ρ(d)) with d its distance to the nearest earlier
   view that also voted there (first voter weight 1). Score = Σ weights.
b) *adjacent-consistency*: a view's vote counts only if a trajectory
   neighbour (step ± 1, or ± 2 on val) also voted within the same voxel or a
   6-neighbour; score = number of consistent votes.
c) *recursive log-odds*: sequential update L ← L + (1 − ρ(d_{j,j−1})) · l_j,
   l_j = +a for a vote, −b for a free-space traversal (view's predicted depth
   beyond the voxel by ≥ 0.2 m), 0 otherwise; (a, b) on val from
   {(1, 0.5), (1, 1), (1, 2)}; score = L.
All three are rankings on the same voxels as the baseline; read at frac 0.25
and at the val-chosen fraction. **Key question**: slope = F1(all) − F1(4) of
the variant vs the baseline under the fixed reference at the top quarter,
where the baseline already saturates: "continuity breaks the headline" if
slope(variant) − slope(baseline) ≥ +0.03 **and** F1(all) − baseline ≥ +0.03,
both sets. Otherwise reported as tie / no.

## E110 — top-k candidates per ray
From the flat posterior heads (T from E21), keep the k highest-mass local
modes per ray (k ∈ {1, 2, 3, 5}; k = 1 is B_argmax), each carrying its mass;
fuse as Σ mass per voxel (`topk_mass`) and as number of rays whose top-k
contains the voxel (`topk_count`); candidate set = union of the top-k
points. Compared with B_argmax (point) and Stage D's C_on_B / C_grid (full
distribution) at the val-chosen fraction and k. **Verdict**: meaningful if
Δ vs B_argmax ≥ 0.03 on both sets at N = all (fixed ref); the position between
point and distribution is reported as the curve over k.

Interactions (E108 × E109) are not pre-registered and, if run, are labelled
post hoc.
