# Pre-registered criteria — E102 baseline-aware support

Written 2026-09-21 before any E102 number was computed. E101 showed that an
added view is precise where it overlaps an earlier view and imprecise where it
is alone, and E100 (c) that views < 0.5 m apart share their errors (r ≈ 0.9)
while views ≥ 1.5 m apart do not (r ≈ 0.4). E102 tests the one remedy this
selects: **count agreement only between views far enough apart**.

## Method (one knob, no training)
On the same voxels and positions as the Stage A pipeline (A_point, 0.1 m
voxels), rank voxels by
- `support`   number of contributing points (Stage A; the control),
- `views`     number of distinct contributing views,
- `cells<Δ>`  number of distinct Δ-long cells along the trajectory occupied by
              contributing views (a fixed-grid clustering of the views that
              agree on the voxel), Δ ∈ {0.3, 0.6, 1.0, 1.5} m, ties broken by
              point count,
- `span`      distance between the two farthest contributing views (m),
              ties broken by point count.
Keep the top fraction f ∈ {0.5, 0.25, 0.1}; the headline fraction is 0.25 (Stage
A's). Fixed reference (GT of every step); the subset reference is recorded too.
N ∈ {1, 2, 4, 8, 16, all}, subset seeds 0–2. Δ (and, separately, f) is chosen on
the val scenes by F1@0.2 at N = all; test is read once at that choice.

## Verdict
- **gain(N)** = F1@0.2(cells<Δ>) − F1@0.2(support), paired over the 39 test
  sequences, same fraction, same N, fixed reference.
- *meaningful* if gain(all) ≥ 0.03 and the bootstrap 95 % CI excludes 0, on
  **both** r2 and r8; *partial* if one; *no effect* if |gain| < 0.03.
- Also reported: gain at N = 4, 8, 16 (a baseline-aware rule cannot help at
  N ≤ 2 where every voxel has at most two views); precision and recall
  separately; whether the top-25 % curve of cells<Δ> is non-decreasing in N.
- `views` and `span` are reported beside cells<Δ> so the reader can see whether
  the gain comes from the distance rule or from counting views instead of
  points. If `views` alone gives the same gain (within 0.03), the distance rule
  has added nothing and that is the result.
