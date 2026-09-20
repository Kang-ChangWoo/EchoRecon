# Pre-registered criteria — E101 marginal-view analysis (why precision falls with N)

Written 2026-09-21 before any E101 number was computed. E100 established (fixed
reference, both observation sets, every scene and subset seed) that F1 of the
support-ranked cloud rises and saturates while **precision falls monotonically
with N at every operating point** (all voxels .51→.35, top 25 % .69→.60).
E101 measures *how* an added view lowers precision, on the released r2 and r8
predictions, fixed reference, no training.

## Measurements (per sequence, views added in the evenly-spread order that
## Stage A/D use for N = 1, 2, 4, 8, 16, all; then in a random order as a control)

For the k-th added view, relative to the cloud of the views already included:
1. **new-voxel precision**: of the voxels the view creates that no earlier view
   occupied, the fraction within 0.2 m of the reference;
2. **overlap fraction**: the fraction of its points that land in voxels an
   earlier view already occupied;
3. **precision of its overlapping points** vs **of its new points**;
4. **distance to the nearest already-included view** (m), and the
   **viewing-angle difference** to that view at the view's median depth;
5. **error class of its wrong new voxels**: near miss (0.2–0.5 m from the
   reference), displaced (0.5–1 m), hallucinated (> 1 m).

## Hypotheses and thresholds (fixed before the run)

- **H1 (neighbour redundancy).** New-voxel precision of an added view
  decreases as its distance to the nearest included view decreases:
  Spearman(distance, new-voxel precision) ≥ +0.3 over all (sequence, k) pairs,
  and the mean new-voxel precision of views added at < 0.5 m is lower than at
  ≥ 1.5 m by ≥ 0.05. If both hold: "added neighbours are the precision leak".
- **H2 (blur, not hallucination).** ≥ 60 % of the wrong new voxels are near
  misses (0.2–0.5 m). If instead ≥ 40 % are > 1 m: the leak is hallucinated
  geometry, which points to a different remedy (free space / confidence)
  than blur does (depth precision).
- **H3 (overlap is where agreement lives).** Precision of overlapping points ≥
  precision of new points + 0.10 at every k ≥ 2. This is the mechanism by which
  support ranking rescues F1; if it fails, support ranking works for another
  reason.
- Random-order control: H1 must also hold when views are added in a random
  order (otherwise it is an artefact of the spread ordering).

## Reporting
Per observation set (r2, r8), per scene mean ± sd over sequences, pooled
Spearman with n. No method is proposed by E101; it decides which remedy family
the next step tests. Differences inside 0.03 (precision) are ties.
