#!/usr/bin/env python3
"""Fuse the predicted ERP depths of a sequence at known poses and score the 3D model.

Reference (diagnostic only, see DIRECTION.md): the ground-truth depths of the
same steps, unprojected at the same poses and voxel-averaged.

Voxel positions are always the plain average of the points that fall in the
voxel, so a rule changes only the *ranking* of voxels, never where they are.
Four rankings are scored on the same voxels:

  uniform       number of contributing points, i.e. the cross-view support
  range_prior   sum of a fixed decreasing function of range: the trivial
                baseline. A confidence that does not beat this has reduced to a
                distance prior.
  model_std     sum of 1 / (eps + observation-dropout spread), when the
                prediction file carries one
  oracle        the voxel's true distance to the reference: the ceiling any
                ranking could reach

For each, the top-fraction curve (keep the best f of the voxels). The recovery
rate at fraction f is

    (uniform_acc - method_acc) / (uniform_acc - oracle_acc)

so 0 means the ranking is no better than counting points and 1 means it is as
good as knowing the answer.

accuracy = mean nearest distance predicted -> reference and the fraction within
tau; completeness = reference -> predicted. The two are always reported
separately. Writes outputs/fusion/<mode>/<scene>/<seq><tag>.json and a top-down
picture.

    python src/eval_fusion.py --scene apartment_2 --seq seq_0000 --mode r2
    python src/eval_fusion.py --scene apartment_2 --seq seq_0000 --n-views 8 --tag _N8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from data import Sequence  # noqa: E402
from erp import quat_to_R, ray_dirs, to_radial  # noqa: E402
from fuse import TSDF, WEIGHTS, accuracy_completeness, voxel_downsample  # noqa: E402

FRACS = (0.75, 0.5, 0.25, 0.1)


def resize_nearest(d: np.ndarray, hw):
    H, W = d.shape
    r = (np.arange(hw[0]) * H // hw[0]); c = (np.arange(hw[1]) * W // hw[1])
    return d[r][:, c]


def unproject_res(depth, pose, dirs, max_depth, stride, kind="face"):
    depth = to_radial(depth, dirs, kind)
    d = depth[::stride, ::stride]; dd = dirs[::stride, ::stride]
    valid = np.isfinite(d) & (d > 0) & (d < max_depth)
    P = dd[valid] * d[valid][:, None]
    R = quat_to_R(pose["rotation"])
    return P @ R.T + np.asarray(pose["position"])[None, :], valid, d[valid], dd[valid] @ R.T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--trunc", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=2, help="pixel stride on the 256x512 prediction")
    ap.add_argument("--convention", default="right0")
    ap.add_argument("--depth-kind", default="face", choices=["face", "radial"],
                    help="what the stored/predicted depth means; the base model outputs per-face "
                         "cubemap z-depth ('face'), converted to radial before unprojection")
    ap.add_argument("--step-stride", type=int, default=1, help="fuse every k-th step")
    ap.add_argument("--n-views", type=int, default=0,
                    help="fuse this many steps, spread evenly over the sequence (0 = all), so the "
                         "recovery rate can be read against the number of views")
    ap.add_argument("--tsdf", action="store_true", help="also extract a TSDF surface (slow)")
    ap.add_argument("--tag", default="", help="suffix on the output file name")
    a = ap.parse_args()
    a.pred_dir = a.pred_dir or REPO / "outputs" / "pred" / a.mode
    a.out_dir = a.out_dir or REPO / "outputs" / "fusion" / a.mode
    z = np.load(a.pred_dir / a.scene / f"{a.seq}.npz")
    pred = z["pred"].astype(np.float32); steps = z["steps"].tolist(); meta = json.loads(str(z["meta"]))
    std = z["std"].astype(np.float32) if z["std"].size else None
    if a.step_stride > 1:
        sel = slice(None, None, a.step_stride)
        pred, steps, std = pred[sel], steps[sel], (std[sel] if std is not None else None)
    if a.n_views and a.n_views < len(steps):
        idx = np.linspace(0, len(steps) - 1, a.n_views).astype(int)
        pred, steps = pred[idx], [steps[i] for i in idx]
        std = std[idx] if std is not None else None
    meta.update(step_stride=a.step_stride, n_views=len(steps), voxel=a.voxel, tau=a.tau,
                stride=a.stride, convention=a.convention, depth_kind=a.depth_kind)
    S = Sequence(a.scene, a.seq)
    H, W = pred.shape[1:]
    dirs = ray_dirs(H, W, a.convention)

    # reference and per-step ERP error
    ref, mae = [], []
    for k, i in enumerate(steps):
        g = resize_nearest(S.gt_depth(i, a.depth_kind), (H, W))
        m = np.isfinite(g) & (g > 0) & (g < a.max_depth)
        mae.append(float(np.abs(pred[k][m] - g[m]).mean()))
        ref.append(unproject_res(g, S.pose(i), dirs, a.max_depth, a.stride, a.depth_kind)[0])
    ref_cloud, _ = voxel_downsample(np.concatenate(ref), a.voxel / 2)
    res = dict(meta=meta, n_steps=len(steps), voxel=a.voxel, tau=a.tau, erp_mae_m=float(np.mean(mae)),
               ref_points=int(len(ref_cloud)))

    # per-step clouds, voxelised like the fused model so the two are comparable
    per, clouds = [], []
    for k, i in enumerate(steps):
        P, valid, d, dw = unproject_res(pred[k], S.pose(i), dirs, a.max_depth, a.stride, a.depth_kind)
        clouds.append((P, d, dw, np.asarray(S.pose(i)["position"], float),
                       (std[k][::a.stride, ::a.stride][valid] if std is not None else None)))
        per.append(accuracy_completeness(voxel_downsample(P, a.voxel)[0], ref_cloud, a.tau))
    res["single_step"] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}

    # one set of voxels, several rankings
    allP = np.concatenate([c[0] for c in clouds])
    vox, _ = voxel_downsample(allP, a.voxel)                      # positions: plain average
    key = np.floor(allP / a.voxel).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    rules = {"uniform": lambda d, s=None: np.ones_like(d, np.float32),
             "range_prior": lambda d, s=None: WEIGHTS["range_prior"](d)}
    if std is not None:
        rules["model_std"] = lambda d, s=None: (1.0 / (0.05 + s)).astype(np.float32)
    scores = {name: np.bincount(inv, weights=np.concatenate([fn(c[1], c[4]) for c in clouds]),
                                minlength=len(vox)) for name, fn in rules.items()}
    d_ref, _ = cKDTree(ref_cloud).query(vox, k=1, workers=-1)      # the ranking that knows the answer
    scores["oracle"] = -d_ref

    res["fused"] = {}
    for name, sc in scores.items():
        order = np.argsort(-sc)
        r = {"all": accuracy_completeness(vox, ref_cloud, a.tau), "kept": {}}
        for f in FRACS:
            keep = order[: max(1, int(f * len(order)))]
            r["kept"][str(f)] = accuracy_completeness(vox[keep], ref_cloud, a.tau)
        res["fused"][name] = r
    res["recovery"] = {}
    for name in scores:
        if name in ("uniform", "oracle"):
            continue
        res["recovery"][name] = {}
        for f in FRACS:
            u = res["fused"]["uniform"]["kept"][str(f)]["acc"]
            o = res["fused"]["oracle"]["kept"][str(f)]["acc"]
            m = res["fused"][name]["kept"][str(f)]["acc"]
            res["recovery"][name][str(f)] = float((u - m) / (u - o)) if u > o else float("nan")

    if a.tsdf:
        lo = allP.min(0) - 0.5; hi = allP.max(0) + 0.5
        tsdf = TSDF(lo, hi, a.voxel, a.trunc)
        for P, d, dw, origin, s in clouds:
            tsdf.integrate(origin, dw, d, np.ones_like(d, np.float32))
        surf, _ = tsdf.surface_points(min_w=1.0)
        res["tsdf_uniform"] = accuracy_completeness(surf, ref_cloud, a.tau)
        res["tsdf_points"] = int(len(surf))

    ss = res["single_step"]
    print(f"[single step ] N={len(steps)} acc {ss['acc']:.3f} m ({100*ss['acc_frac']:.1f}% <{a.tau} m) "
          f"comp {ss['comp']:.3f} | ERP MAE {res['erp_mae_m']:.3f} m")
    for name in scores:
        r = res["fused"][name]
        kept = "  ".join(f"top{int(100*f)}% {r['kept'][str(f)]['acc']:.2f}/{100*r['kept'][str(f)]['acc_frac']:.0f}%"
                         for f in FRACS)
        rec = ("  recovery " + " ".join(f"{res['recovery'][name][str(f)]:+.2f}" for f in FRACS)) if name in res["recovery"] else ""
        print(f"[{name:11s}] all {r['all']['acc']:.3f}/{100*r['all']['acc_frac']:.0f}% comp {r['all']['comp']:.3f} | {kept}{rec}")
    out = a.out_dir / a.scene / f"{a.seq}{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    top = np.argsort(-scores["uniform"])[: max(1, len(vox) // 4)]
    for ax, P, t in ((axes[0], ref_cloud, "reference (GT depth fused)"),
                     (axes[1], vox, "prediction fused, all voxels"),
                     (axes[2], vox[top], "prediction fused, top 25% by support")):
        ax.scatter(P[:, 0], -P[:, 2], s=0.3, c=P[:, 1], cmap="viridis",
                   vmin=allP[:, 1].min(), vmax=allP[:, 1].max())
        ax.set_aspect("equal"); ax.set_title(f"{a.scene}/{a.seq} {t}", fontsize=8); ax.axis("off")
    traj = np.array([c[3] for c in clouds])
    for ax in axes[1:]:
        ax.plot(traj[:, 0], -traj[:, 2], "r.-", ms=3, lw=0.8)
    fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=130)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
