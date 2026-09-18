#!/usr/bin/env python3
"""Steps 2/3: fuse the predicted ERP depths of a sequence at known poses and score the 3D model.

Reference: the ground-truth depths of the same steps, unprojected at the same
poses and voxel-averaged (the "true surface" a perfect predictor would give).

Reported per sequence:
  single-step   mean over steps of the accuracy of one step's predicted cloud
                against the reference (what the base model gives on its own)
  fused/<w>     the weighted fusion of all steps under weight rule <w>
                (uniform, range_prior, and, when the prediction file carries
                observation-dropout std, model_std = 1 / (eps + std)), as a
                weighted voxel average and as a weighted TSDF surface
  per-step ERP  MAE of the prediction against the GT ERP depth (256x512)

accuracy = mean nearest distance predicted -> reference and the fraction within
tau; completeness = reference -> predicted. Writes outputs/fusion/<scene>/<seq>.json
and a top-down scatter PNG.

    python src/eval_fusion.py --scene apartment_2 --seq seq_0000
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
from data import Sequence  # noqa: E402
from erp import quat_to_R, ray_dirs  # noqa: E402
from fuse import TSDF, WEIGHTS, accuracy_completeness, voxel_downsample  # noqa: E402


def resize_nearest(d: np.ndarray, hw):
    H, W = d.shape
    r = (np.arange(hw[0]) * H // hw[0]); c = (np.arange(hw[1]) * W // hw[1])
    return d[r][:, c]


def unproject_res(depth, pose, dirs, max_depth, stride):
    d = depth[::stride, ::stride]; dd = dirs[::stride, ::stride]
    valid = np.isfinite(d) & (d > 0) & (d < max_depth)
    P = dd[valid] * d[valid][:, None]
    R = quat_to_R(pose["rotation"])
    return P @ R.T + np.asarray(pose["position"])[None, :], valid, d[valid], dd[valid] @ R.T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--pred-dir", type=Path, default=REPO / "outputs" / "pred")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "fusion")
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--trunc", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=2, help="pixel stride on the 256x512 prediction")
    ap.add_argument("--convention", default="right0")
    a = ap.parse_args()
    z = np.load(a.pred_dir / a.scene / f"{a.seq}.npz")
    pred = z["pred"].astype(np.float32); steps = z["steps"].tolist(); meta = json.loads(str(z["meta"]))
    std = z["std"].astype(np.float32) if z["std"].size else None
    S = Sequence(a.scene, a.seq)
    H, W = pred.shape[1:]
    dirs = ray_dirs(H, W, a.convention)

    # reference and per-step ERP error
    ref, mae = [], []
    for k, i in enumerate(steps):
        g = resize_nearest(S.gt_depth(i), (H, W))
        m = np.isfinite(g) & (g > 0) & (g < a.max_depth)
        mae.append(float(np.abs(pred[k][m] - g[m]).mean()))
        ref.append(unproject_res(g, S.pose(i), dirs, a.max_depth, a.stride)[0])
    ref_cloud, _ = voxel_downsample(np.concatenate(ref), a.voxel / 2)
    res = dict(meta=meta, n_steps=len(steps), voxel=a.voxel, tau=a.tau, erp_mae_m=float(np.mean(mae)),
               ref_points=int(len(ref_cloud)))

    # single-step clouds, voxelised like the fused model so the two are comparable
    # (voxel averaging thins the dense near field and keeps every far outlier, so
    # raw clouds and voxelised clouds are not on the same footing)
    per, clouds = [], []
    for k, i in enumerate(steps):
        P, valid, d, dw = unproject_res(pred[k], S.pose(i), dirs, a.max_depth, a.stride)
        clouds.append((P, d, dw, np.asarray(S.pose(i)["position"], float), (std[k][::a.stride, ::a.stride][valid] if std is not None else None)))
        per.append(accuracy_completeness(voxel_downsample(P, a.voxel)[0], ref_cloud, a.tau))
    res["single_step"] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}

    # fused, per weight rule
    lo = np.concatenate([c[0] for c in clouds]).min(0) - 0.5
    hi = np.concatenate([c[0] for c in clouds]).max(0) + 0.5
    rules = dict(WEIGHTS)
    if std is not None:
        rules["model_std"] = lambda depth, s=None, **_: (1.0 / (0.05 + s)).astype(np.float32)
    res["fused"] = {}
    allP = np.concatenate([c[0] for c in clouds])
    for name, fn in rules.items():
        w = np.concatenate([fn(c[1], s=c[4]) for c in clouds])
        vox, wsum = voxel_downsample(allP, a.voxel, w)
        r = {"voxel_avg": accuracy_completeness(vox, ref_cloud, a.tau)}
        # the weights rank voxels; keeping the top fraction trades completeness for
        # accuracy, and a weight rule is only informative if its ranking beats the
        # others' at the same kept fraction
        order = np.argsort(-wsum)
        r["kept"] = {}
        for frac in (0.75, 0.5, 0.25):
            keep = order[: max(1, int(frac * len(order)))]
            r["kept"][str(frac)] = accuracy_completeness(vox[keep], ref_cloud, a.tau)
        tsdf = TSDF(lo, hi, a.voxel, a.trunc)
        for P, d, dw, origin, s in clouds:
            tsdf.integrate(origin, dw, d, fn(d, s=s))
        surf, _ = tsdf.surface_points(min_w=1.0)
        r["tsdf"] = accuracy_completeness(surf, ref_cloud, a.tau)
        r["tsdf_points"] = int(len(surf))
        res["fused"][name] = r
        kept = " ".join(f"top{int(100*float(f))}%: acc {v['acc']:.2f}/{100*v['acc_frac']:.0f}% comp {v['comp']:.2f}" for f, v in r["kept"].items())
        print(f"[{name:11s}] all: acc {r['voxel_avg']['acc']:.3f} m ({100*r['voxel_avg']['acc_frac']:.1f}% <{a.tau} m) comp {r['voxel_avg']['comp']:.3f} | "
              f"{kept} | tsdf acc {r['tsdf']['acc']:.3f} ({100*r['tsdf']['acc_frac']:.1f}%) comp {r['tsdf']['comp']:.3f}")
    ss = res["single_step"]
    print(f"[single-step ] acc {ss['acc']:.3f} m ({100*ss['acc_frac']:.1f}% <{a.tau} m) comp {ss['comp']:.3f} | ERP MAE {res['erp_mae_m']:.3f} m")
    out = a.out_dir / a.scene / f"{a.seq}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))

    # top-down picture: reference vs fused (uniform)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = np.concatenate([WEIGHTS["uniform"](c[1]) for c in clouds])
    vox, _ = voxel_downsample(allP, a.voxel, w)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, P, t in ((axes[0], ref_cloud, "reference (GT depth fused)"), (axes[1], vox, "prediction fused (uniform)")):
        ax.scatter(P[:, 0], -P[:, 2], s=0.3, c=P[:, 1], cmap="viridis", vmin=lo[1], vmax=hi[1])
        ax.set_aspect("equal"); ax.set_title(f"{a.scene}/{a.seq}: {t}", fontsize=9); ax.axis("off")
    traj = np.array([c[3] for c in clouds]); axes[1].plot(traj[:, 0], -traj[:, 2], "r.-", ms=3, lw=0.8)
    fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=130)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
