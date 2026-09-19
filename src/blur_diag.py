#!/usr/bin/env python3
"""Step 2: is the prediction a blurred version of the truth, and how wide is the blur?

No training. Two questions, both answered on the model's own output space (the
per-face depth it was trained on), so nothing here depends on the unprojection.

1. Blur width. The ground-truth ERP depth is blurred by a Gaussian of angular
   width sigma (degrees of arc, so the horizontal kernel widens towards the
   poles where a column spans less angle) and the prediction is compared with
   it. The sigma that minimises |prediction - blur(GT)| is how much the
   prediction has smoothed the room away. Reported next to:
     - the error against the unblurred truth (sigma = 0),
     - how much blurring alone costs, |blur(GT) - GT|, so it is visible whether
       the prediction is better than the blur it resembles,
     - the same curve for the sequence-mean prediction, the static shell, which
       bounds how much of the fit is just "every step looks alike".

2. Roughness. Mean absolute angular gradient of the prediction, of the truth,
   and of the truth blurred at the fitted sigma. A prediction that is a smoothed
   truth has the roughness of the blurred truth, not of the truth.

    python src/blur_diag.py --mode r2
    python src/blur_diag.py --mode r2 --pred-dir outputs/pred_hop40/r2 --tag _hop40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from data import TEST_SCENES, Sequence, sequences  # noqa: E402
from eval_fusion import resize_nearest  # noqa: E402

SIGMAS_DEG = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
BANDS = 16          # latitude bands sharing one horizontal kernel width


def erp_blur(img: np.ndarray, sigma_deg: float) -> np.ndarray:
    """Gaussian of `sigma_deg` degrees of arc on the sphere, separable and
    latitude-corrected: a column spans cos(elevation) times less angle near the
    poles, so the horizontal kernel is widened there."""
    if sigma_deg <= 0:
        return img
    H, W = img.shape
    s_row = sigma_deg / (180.0 / H)
    out = gaussian_filter1d(img, s_row, axis=0, mode="nearest")
    elev = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    edges = np.linspace(0, H, BANDS + 1).astype(int)
    res = np.empty_like(out)
    for a, b in zip(edges[:-1], edges[1:]):
        c = max(np.cos(elev[a:b]).mean(), 1e-3)
        s_col = sigma_deg / (360.0 / W) / c
        res[a:b] = gaussian_filter1d(out[a:b], s_col, axis=1, mode="wrap")
    return res


def roughness(img: np.ndarray) -> float:
    return float(0.5 * (np.abs(np.diff(img, axis=0)).mean() + np.abs(np.diff(img, axis=1)).mean()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--scenes", nargs="+", default=list(TEST_SCENES))
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--steps-per-seq", type=int, default=8, help="steps sampled evenly per sequence")
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    pred_dir = a.pred_dir or REPO / "outputs" / "pred" / a.mode
    curves = {s: [] for s in SIGMAS_DEG}        # |pred - blur(GT)|
    cost = {s: [] for s in SIGMAS_DEG}          # |blur(GT) - GT|
    shell = {s: [] for s in SIGMAS_DEG}         # |mean pred - blur(GT)|
    rough = {"pred": [], "gt": []}
    rough_blur = {s: [] for s in SIGMAS_DEG}
    n_seq = 0
    for sc in a.scenes:
        for sq in sequences(sc):
            f = pred_dir / sc / f"{sq}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            P = z["pred"].astype(np.float32); steps = z["steps"].tolist()
            S = Sequence(sc, sq)
            pick = np.linspace(0, len(steps) - 1, min(a.steps_per_seq, len(steps))).astype(int)
            Pm = P.mean(0)
            for k in pick:
                p = P[k]
                g = resize_nearest(S.gt_depth(steps[k], "face"), p.shape).astype(np.float32)
                m = np.isfinite(g) & (g > 0) & (g < a.max_depth)
                if m.sum() < 100:
                    continue
                gf = np.where(m, g, np.nan)
                gf = np.where(np.isnan(gf), np.nanmedian(gf), gf)     # fill so the blur is defined
                rough["pred"].append(roughness(p)); rough["gt"].append(roughness(gf))
                for s in SIGMAS_DEG:
                    b = erp_blur(gf, s)
                    curves[s].append(float(np.abs(p[m] - b[m]).mean()))
                    cost[s].append(float(np.abs(b[m] - g[m]).mean()))
                    shell[s].append(float(np.abs(Pm[m] - b[m]).mean()))
                    rough_blur[s].append(roughness(b))
            n_seq += 1
    if not n_seq:
        print("no predictions found"); return 1
    C = {s: float(np.mean(v)) for s, v in curves.items()}
    K = {s: float(np.mean(v)) for s, v in cost.items()}
    Sh = {s: float(np.mean(v)) for s, v in shell.items()}
    Rb = {s: float(np.mean(v)) for s, v in rough_blur.items()}
    best = min(C, key=C.get); best_shell = min(Sh, key=Sh.get)
    print(f"[{a.mode}{a.tag}] {n_seq} sequences, {len(curves[0.0])} frames, predictions from {pred_dir}")
    print(f"{'sigma (deg)':>11s} {'|pred-blur(GT)|':>16s} {'|blur(GT)-GT|':>14s} {'|meanpred-blur(GT)|':>20s}")
    for s in SIGMAS_DEG:
        mark = "  <- best" if s == best else ""
        print(f"{s:11.1f} {C[s]:16.4f} {K[s]:14.4f} {Sh[s]:20.4f}{mark}")
    print(f"\nbest sigma {best:.1f} deg: the prediction is closer to the truth blurred by {best:.1f} deg "
          f"({C[best]:.3f} m) than to the truth itself ({C[0.0]:.3f} m)"
          if best > 0 else f"\nbest sigma 0 deg: the prediction is closest to the unblurred truth")
    print(f"static shell (sequence-mean prediction) is closest at {best_shell:.1f} deg, {Sh[best_shell]:.3f} m")
    rp, rg = float(np.mean(rough["pred"])), float(np.mean(rough["gt"]))
    print(f"roughness (mean |gradient|, m/pixel): prediction {rp:.4f}  truth {rg:.4f}  "
          f"truth blurred at best sigma {Rb[best]:.4f}")
    out = REPO / "outputs" / f"blur_{a.mode}{a.tag}.json"
    out.write_text(json.dumps(dict(mode=a.mode, tag=a.tag, pred_dir=str(pred_dir), n_seq=n_seq,
                                   n_frames=len(curves[0.0]), sigmas_deg=list(SIGMAS_DEG),
                                   pred_vs_blur=C, blur_cost=K, shell_vs_blur=Sh,
                                   best_sigma_deg=best, best_sigma_shell_deg=best_shell,
                                   roughness=dict(pred=rp, gt=rg, blurred_gt=Rb)), indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
