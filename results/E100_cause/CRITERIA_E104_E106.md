# Pre-registered criteria — E104 lobe weighting, E105 posterior fusion under the fixed reference, E106 learned per-voxel evidence

Written 2026-09-21 before any of the three was computed. The owner asked that
the remaining catalogue items — "the region each observation does well"
(lobe), the depth distribution, and the graph — each be tested as fast as
possible on the present data, in that order of cost. The wide-trajectory
re-render (the recommended first step) cannot run on this machine (no
habitat-sim) and is [미확인] until the render server is used.

Common rules: released predictions, fixed reference, Stage A voxels, headline
operating point = support top 25 %, band 0.03, paired bootstrap CI over the 39
test sequences, both r2 and r8; every hyper-parameter chosen on the val scenes.

## E104 — lobe weighting ("잘하는 영역")
*Prerequisite measurement (val and test, reported separately).* Per-ray wrong
rate (|error| > 0.2 m) and MAE of the released model by azimuth relative to
the observation's forward axis (12 bins of 30°) and by elevation (6 bins of
30°), pooled over sequences. The lobe is *real* if, on val, the best and worst
azimuth bins differ in wrong rate by ≥ 0.15 (r2) — and the elevation curve is
reported beside it (catalogue prescription 2's precondition). If the azimuth
curve is flat (< 0.15), E104's ranking is not run and that is the result.
*Ranking.* Per-ray weight w = 1 − wrong_rate(azimuth bin, elevation bin)
estimated on **val**; voxels ranked by Σ w (`lobe_sum`) and by Σ w with the
azimuth dimension collapsed (`elev_only`, the isotropic-in-azimuth control)
and by Σ w with elevation collapsed (`azim_only`). Verdict: gain over
`support` at N = all ≥ 0.03, CI above 0, both sets; *and* `lobe_sum` must beat
`elev_only` by ≥ 0.03, otherwise the gain is not the lobe's.

## E105 — posterior fusion under the fixed reference ("분포")
Stage D v3 re-run with `--ref fixed` (`results/E30_stage_d/{r2,r8}_v3fixed`,
`*_v3fixed_val`), everything else as v3 (band interp, val-fitted T, one box).
Kept fraction chosen on val per method (`src/select_frac.py`). Verdict as in
`results/E30_stage_d/CRITERIA.md`'s secondary rule, with the fixed reference:
C (C_on_B or C_grid, whichever is better on val) vs A_point: slope(C) −
slope(A) ≥ +0.03 and C(all) − A(all) ≥ +0.03 → "the distribution restores the
value of extra views"; |both| < 0.03 → tie; C below A by ≥ 0.03 → no. D and E
are read against B at the same voxels / same count, as before.

## E106 — learned per-voxel evidence (the fast test of the graph's premise)
The bipartite graph's promise is that edges carrying per-observation
information (relative angle, baseline, confidence, free space) let the fusion
tell right voxels from wrong ones better than counting. The fastest test of
that premise: a small MLP on per-voxel aggregates of exactly those edge
messages — point count, distinct views, Δ-cell counts (0.3/0.6/1.0), span,
free-space contradiction count, viewing-angle spread, mean / max / min ray
confidence, mean / min range, mean incidence cosine, mean |elevation|, N —
trained to predict "within 0.2 m of the reference" on the **val** scenes (37
sequences, N = all and N = 4 rows) and used as a ranking on the **test**
scenes. Controls: `support` (count) and a logistic model on count alone.
Verdict: gain over `support` at N = all ≥ 0.03, CI above 0, both sets; AUROC
against the count-only model is reported. If it fails, the information the
graph would route does not exist in these predictions on this data, and the
full bipartite model is not built; if it passes, the full model is the next
step and its control is fully-connected attention. This is explicitly a proxy,
not the graph itself.
