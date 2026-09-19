# Working direction

Set by the project owner. Every decision is judged by whether it raises the
chance of acceptance; the hypothesis is kept simple. The word "first" is not
used anywhere in this work: analytic and neural audio-to-geometry methods
already exist (DAPS, SonoNERFs, and the analytic line).

## Hypothesis

A single echo observation does not carry enough information to place a surface,
so the predicted depth collapses towards an average shell. A surface is real
where several sufficiently separated views agree. What this work measures is
how much of that agreement can be recovered, and what the ceiling of
audio-only 3D is.

## Established so far (README step 2, before the reproduction)

- Uniform fusion is worse than a single step (accuracy 0.52 -> 0.64 m).
  Keeping the quarter of voxels with the most cross-view support gives 0.36 m,
  while completeness degrades 0.40 -> 0.83 m.
- Uniform, range-prior and model-std weights do not change the support ranking:
  the weight sum is dominated by how many points fall in the voxel.
- The dropout spread for `r2` is a weak signal because only one of the two
  observations can be dropped (kmax = 1). To be re-measured on `r6` / `r8`.
- Each step's prediction is a smooth shell around the receiver; the shells pile
  into a blob along the trajectory.

(The numbers above predate the per-face depth-convention fix and are being
re-measured; the qualitative statements have not changed.)

## Order of work

No step's conclusion is drawn before the previous step's result exists.

1. **Oracle ceiling.** Rank the fused voxels by their true distance to the
   reference and draw that ranking's top-fraction curve beside the support
   curve. Report the recovery rate (method − uniform) / (oracle − uniform) per
   number of views N.
2. **Blur diagnosis (no training).** Test whether the prediction is closer to a
   heavily blurred ground truth than to the ground truth itself, and whether the
   shell thickness moves with the input STFT hop (`--hop` sweep).
3. **Support, refined.** Restrict the comparison views to those at least Δ
   apart and compute Spearman(support, |error|) per Δ, to expose the shared bias
   of neighbouring steps 0.15 m apart. Classify agreement with earlier views in
   three states: supported / contradicted (earlier view saw free space through
   it) / unobserved. Unobserved is not penalised. Offline by default, causal as
   an ablation.
4. **Learned confidence.** A small head predicts "is this point right" against
   the true error. Inputs: range, elevation, incidence angle, dropout spread,
   leave-one-step-out support, Δ-restricted agreement. Validation: the
   range-prior-only baseline, a shuffled-confidence control, injected corruption
   of some views, sparsification / AUSE, Spearman. Twelve training scenes only,
   so the head stays small.
5. **Distribution head** (after 3 and 4). Replace the base model's point depth
   with a per-ray depth-bin distribution and accumulate the likelihood volume in
   log-odds. The base checkpoint is not pinned before its remote state has been
   checked.

## Evaluation

- The current reference (the same steps' ground-truth depth, fused) is kept for
  diagnosis only.
- For the paper, build a ground-truth occupancy from the scene mesh and report
  IoU / Chamfer / F1, following the same protocol as DAPS where possible.
- Accuracy and completeness are always reported separately, and the treatment of
  unobserved regions is stated.

## Rules

- The success criteria X and Y and the prior predictions are the owner's to set.
  They are not filled in here.
- The held-out split stands (3 validation scenes, 3 test scenes). Choices are
  made on validation; test is looked at once.
- Every output records hop, observation set, voxel size and tau as metadata.
  The base model's code is imported from the sibling checkout, never copied.
- Every number is reported with four things: why, what was done, the result, and
  the reading. Anything not checked is marked 미확인 (unverified).
- The base model's code and text are not sent to external services.
- Scope is not reduced for schedule reasons; only research risk is reported.

## Undecided (owner's call; implemented as switches)

- Whether the ground truth moves to a mesh-based occupancy.
- Whether a supervised confidence is acceptable.
- The scope of the distribution head.
- The blanks in the paper's opening sentence ("collapses to ___", "___ views
  agree").
