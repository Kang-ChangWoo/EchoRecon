#!/usr/bin/env python3
"""Does the prediction move with the receiver, or is it a static shell?

Fusing several posed predictions can only help if the predictions differ with
the pose in the way the true depths do. This measures that directly on a
sequence, in the ERP frame (both predicted and true depths are listener-centred
and already share the frame at every step):

  moved        mean |d_i - d_{i+1}| between consecutive steps (0.15 m apart),
               for the prediction and for the ground truth
  spread       per-pixel standard deviation across the steps of the sequence,
               averaged over pixels, again for both
  residual r   correlation between (pred_i - mean_pred) and (gt_i - mean_gt)
               over pixels: whether the part of the prediction that *does*
               change tracks the part of the truth that changes
  static-shell the error of replacing every prediction by the sequence's mean
  penalty      prediction, against the error of the prediction itself

If the prediction's spread is far below the truth's and the residual
correlation is near zero, each step is close to one fixed shell and no fusion
rule can recover the room from it.

    python src/motion_diag.py --mode r2 --scenes apartment_2 frl_apartment_5 office_4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from data import TEST_SCENES, Sequence, sequences  # noqa: E402
from eval_fusion import resize_nearest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--scenes", nargs="+", default=list(TEST_SCENES))
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    pred_dir = REPO / "outputs" / "pred" / a.mode
    rows = []
    for sc in a.scenes:
        for sq in sequences(sc):
            f = pred_dir / sc / f"{sq}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            P = z["pred"].astype(np.float32)                      # (T, H, W)
            steps = z["steps"].tolist()
            S = Sequence(sc, sq)
            G = np.stack([resize_nearest(S.gt_depth(i, "face"), P.shape[1:]) for i in steps]).astype(np.float32)
            M = np.isfinite(G) & (G > 0) & (G < a.max_depth)
            keep = M.all(0)                                        # pixels valid at every step
            if keep.sum() < 100:
                continue
            p, g = P[:, keep], G[:, keep]                          # (T, N)
            pos = np.array([S.pose(i)["position"] for i in steps])
            travel = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
            dp = float(np.abs(np.diff(p, axis=0)).mean())
            dg = float(np.abs(np.diff(g, axis=0)).mean())
            sp = float(p.std(0).mean()); sg = float(g.std(0).mean())
            pc = p - p.mean(0); gc = g - g.mean(0)
            denom = np.sqrt((pc ** 2).sum() * (gc ** 2).sum())
            r = float((pc * gc).sum() / denom) if denom > 0 else np.nan
            err = float(np.abs(p - g).mean())
            err_mean_shell = float(np.abs(p.mean(0)[None, :] - g).mean())
            rows.append(dict(scene=sc, seq=sq, steps=len(steps), travel_m=travel,
                             moved_pred=dp, moved_gt=dg, spread_pred=sp, spread_gt=sg,
                             residual_r=r, mae=err, mae_static_shell=err_mean_shell))
            print(f"[{sc}/{sq}] {len(steps):3d} steps, {travel:5.2f} m travelled | moved {dp:.3f} vs {dg:.3f} m | "
                  f"spread {sp:.3f} vs {sg:.3f} m | residual r {r:+.3f} | MAE {err:.3f} vs static shell {err_mean_shell:.3f}",
                  flush=True)
    if not rows:
        print("no predictions found"); return 1
    A = {k: float(np.mean([r[k] for r in rows])) for k in
         ("travel_m", "moved_pred", "moved_gt", "spread_pred", "spread_gt", "residual_r", "mae", "mae_static_shell")}
    print(f"\n[{a.mode}, {len(rows)} sequences] moved {A['moved_pred']:.3f} vs GT {A['moved_gt']:.3f} m "
          f"({100*A['moved_pred']/A['moved_gt']:.0f}% of the truth's step-to-step change)")
    print(f"  spread across steps {A['spread_pred']:.3f} vs GT {A['spread_gt']:.3f} m "
          f"({100*A['spread_pred']/A['spread_gt']:.0f}%)   residual correlation {A['residual_r']:+.3f}")
    print(f"  MAE {A['mae']:.3f} m; replacing every step by the sequence mean prediction: {A['mae_static_shell']:.3f} m "
          f"({A['mae_static_shell'] - A['mae']:+.3f})")
    out = a.out or REPO / "outputs" / f"motion_{a.mode}.json"
    out.write_text(json.dumps(dict(mode=a.mode, mean=A, sequences=rows), indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
