# Pre-registered verdict criteria for Stage D v3

Written 2026-09-20 before any v3 run was started, and committed so the hash
proves it. Not to be changed after results are read.

**Honesty note.** Earlier runs (`r2`, `r8` at 0e85bc7/48f5398 with whole-bin
band rounding, temperature 1 and a reference-derived grid box; `r2_v2`, `r8_v2`
whose recorded commit a57f2cd does not match the code that ran, since band
interpolation was uncommitted at the time) have already been seen. Those numbers
cannot be un-seen. What is fixed here is the *rule* by which v3 is judged, so
that v3 is not read to fit a conclusion. The thresholds below are chosen from
the measured subset-to-subset spread, not from any v2 number.

## Runs that count

- Heads: `posterior_r2`, `posterior_r8` (best by val KL = epoch 2, the first
  unfrozen epoch, confirmed from `outputs/posterior/*/history.json`), **and**
  one retrain per observation set with a longer head-only warm-up and a lower
  backbone learning rate (`--warmup-epochs 4 --lr 5e-5 --epochs 12`, otherwise
  identical), selected by the same rule (val KL). Both are reported; the
  retrain exists to answer "was the head under-trained", not to pick the better
  of two on test.
- Stage D v3: `src/stage_d.py` at the commit named in each result directory's
  `git_commit.txt`, `--split test` for numbers, `--split val` only to choose
  the kept fraction (`src/select_frac.py`). Band rounding `interp`, temperature
  from `results/E21_posterior_rays/<run>_test.json` (fitted on validation).
- Ablation: `r2 --band whole` with everything else as v3, to isolate the
  band-rounding fix.

## Primary verdict: ray level (E21 on the test split)

The claim under test is "preserving the ambiguity keeps the true depth alive
where the point estimate is wrong", so the primary evidence is per ray, not the
3-D F1.

| quantity | yes | weak | no |
|---|---|---|---|
| mode rescue@5 (fraction of wrong-argmax rays with a local maximum within 0.2 m of the truth) | ≥ 20 % | 5–20 % | < 5 % |
| mass within ±0.2 m of the truth when the argmax is wrong | ≥ 30 % | 15–30 % | < 15 % |
| NLL of the learned posterior vs the fake Gaussian at its best sigma | lower by ≥ 0.5 | lower by 0–0.5 | not lower |
| confidence–error Spearman (max softmax vs argmax error) | ≤ −0.5 | −0.3 to −0.5 | > −0.3 |

The fake Gaussian control (same argmax, fixed sigma) must be beaten on rescue
and NLL for any "yes"; it cannot rescue by construction, so rescue is judged on
its absolute value.

## Secondary verdict: view-count curve (F1@0.2 on test at the val-selected kept fraction)

Tie band **0.03 F1**, which is the spread between view subsets measured in
Stage A (seeds 0/1/2: 0.549 / 0.547 / 0.518 at N=4). Differences inside the
band are ties.

- *Shape*: slope = F1(all views) − F1(N=4). "C restores the value of extra
  views" if slope(C) − slope(A) ≥ +0.03 **and** C(all) − A(all) ≥ +0.03. If
  both |differences| < 0.03: tie. If C is below A by ≥ 0.03 at N=all: no.
  C is whichever of C_on_B / C_grid is better *on validation*.
- *Single view*: C_grid − A_point at N=1, same band.
- *Argmax equivalence* (E31): |B − A| at N=1 must be < 0.03 for the B-vs-C
  comparison to be read as "representation, not network".
- D and E are read against B only (same rays); E against
  `B_argmax_matchE*` at the same kept count.

## What is not a criterion

Validation F1 is never used to rank methods. The oracle is a ceiling for the
B-voxel methods only. Metrics at 0.1 m and IoU are not compared for C_grid
when its grid was coarsened (`grid_voxel` > 0.1 in the row).
