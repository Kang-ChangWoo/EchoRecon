# Pre-registered criteria — E111 per-ray reliability head inside the 2-D model (owner's plan A)

Written 2026-09-23 before any E111 number was computed. E106b's post-fusion
ranker (an MLP on per-voxel aggregates of ray geometry; F1 .605 / .627 at the
top quarter, +0.050 / +0.054 over counting) proved the information exists;
E108 proved the far field is wrong in the predictions, not under-voted. Plan
A moves that information into the 2-D model as a per-ray head so the model
can learn it.

## Model
`ReliabilityDepth` wraps the released base model: the decoder feature map x
(before the depth head) feeds (i) the original depth head (unchanged) and
(ii) a new reliability head r = σ(Conv1×1(ReLU(Conv3×3([x, d̂/max_depth,
sin φ, cos φ])))) with d̂ the model's own depth (detached) and φ the pixel
elevation — the two features E106b found dominant, given explicitly so the
head does not have to rediscover them. Target: 1[|d̂ − d| < 0.2 m] per valid
ray (the E106 label at ray level). Loss: BCE with positive-class weight
(1 − p)/p from the training positive rate.
Variants: **H1 head-only** (backbone and depth head frozen; the released
depths are unchanged, so the fusion baseline is exactly A_point) and **H2
joint** (backbone fine-tuned at lr 5e-5 with loss L1(d̂, d) + λ·BCE, λ = 1;
head lr 1e-3; warm-up 2 head-only epochs). 20 epochs, early stop after 6
without val-BCE improvement, best by val BCE. Both r2 and r8. Checkpoints in
`/root/local1/changwoo/echorecon_ckpt/`, mirrored metadata in
`outputs/reliability/`. Trained on the 12 train scenes, val for selection,
test read once.

## Fusion and comparison (fixed reference, top quarter unless stated, N = all; N-curve reported)
On each variant's own point cloud (H1 = base cloud): rankings `count`
(baseline), `rel_sum` (Σ r), `rel_mean` (mean r, ties by count); the E106b
ranker refit on the variant's train-scene voxel features **without** the
head's aggregates (`ranker`), and **with** them (`ranker+head`).
- **Verdict 1 (head beats the post-fusion ranker):** best head ranking
  (rel_sum / rel_mean, chosen on val) − E106b ranker (.605 / .627 on the flat
  cloud; recomputed on the variant's cloud) ≥ 0.03 with paired 95 % CI above 0
  on both r2 and r8. Beating `count` alone is not the bar.
- **Verdict 2 (non-redundancy):** gain(ranker+head) − max(gain(ranker),
  gain(head)) ≥ 0.03 → the two carry different information ("adds up");
  within ±0.03 → the head is the same information moved ("eats"); this is
  reported either way.
- **Verdict 3 (far field, ≥ 4 m):** precision and recall of the kept voxels
  in the ≥ 4 m range bin at fractions {0.5, 0.25, 0.1}. The E110 top-k
  points on the flat head define the reference P/R curve for the far bin
  (r2: recall .083 → precision .576 at B .25; .215 → .453 at B .50; .417 →
  .348 at mass k=2 .25; .749 → .169 at mass k=5 .25; r8 analogous), read as
  the far-bin precision at the far-bin recall the head reaches by
  log-linear interpolation. "Above the curve" if the head's far-bin
  precision exceeds the interpolated value by ≥ 0.05 at its recall on both
  sets; "on the curve" otherwise (then the head only changed the budget).
- **Distribution shift:** the head's target positive rate on train / val /
  test rays and the val vs test BCE and AUROC are recorded; a train–val gap
  of the same size as E106b's (.42–.48 vs .32–.39) is expected and noted, a
  larger test–val gap would be flagged.
Hyper-parameters (which ranking, fraction, λ if swept) on val only.
