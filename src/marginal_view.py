#!/usr/bin/env python3
"""E101: marginal-view analysis — how does one more view lower precision?

Views are added one at a time in three orders (spread = farthest-point order,
random, sequential). For the k-th added view, against the cloud of the views
already included (0.1 m voxels) and the fixed reference (GT of every step):
new-voxel precision, overlap fraction, precision of overlapping vs new points,
distance to the nearest included view, and the error class of the wrong new
voxels (near miss 0.2-0.5 m / displaced 0.5-1 m / hallucinated > 1 m).
Criteria: results/E100_cause/CRITERIA_E101.md.

    python src/marginal_view.py --mode r2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from cause_decomp import pack, voxelize  # noqa: E402
from data import TEST_SCENES, VAL_SCENES, Sequence, sequences  # noqa: E402
from erp import quat_to_R, ray_dirs, to_radial  # noqa: E402
from eval_fusion import resize_nearest  # noqa: E402

CFG = {}


def farthest_order(origins: np.ndarray) -> list[int]:
    T = len(origins)
    order = [0]
    d = np.linalg.norm(origins - origins[0], axis=1)
    for _ in range(T - 1):
        nxt = int(np.argmax(d)); order.append(nxt)
        d = np.minimum(d, np.linalg.norm(origins - origins[nxt], axis=1))
    return order


def one_sequence(job):
    sc, sq = job
    a = CFG; voxel, stride, md, tau = a["voxel"], a["stride"], a["max_depth"], a["tau"]
    pf = Path(a["pred_dir"]) / sc / f"{sq}.npz"
    if not pf.exists():
        return []
    z = np.load(pf); P_face = z["pred"].astype(np.float32); steps = z["steps"].tolist()
    S = Sequence(sc, sq); T = len(steps); H, W = P_face.shape[1:]
    dirs = ray_dirs(H, W, a["convention"]); dirs_s = dirs[::stride, ::stride]
    origins, clouds, med_depth, gts = [], [], [], []
    for k, i in enumerate(steps):
        pose = S.pose(i); R = quat_to_R(pose["rotation"]); o = np.asarray(pose["position"], float); origins.append(o)
        d = to_radial(P_face[k], dirs, "face")[::stride, ::stride]; v = np.isfinite(d) & (d > 0) & (d < md)
        clouds.append((dirs_s[v] * d[v][:, None]) @ R.T + o); med_depth.append(float(np.median(d[v])))
        g = to_radial(resize_nearest(S.gt_depth(i, "face"), (H, W)), dirs, "face")[::stride, ::stride]
        gv = np.isfinite(g) & (g > 0) & (g < md)
        gts.append((dirs_s[gv] * g[gv][:, None]) @ R.T + o)
    origins = np.stack(origins)
    _, ref, _, _ = voxelize(np.concatenate(gts), voxel / 2)
    tree = cKDTree(ref)
    # per-view voxel keys, per-point reference distance (computed once)
    keys = [pack(c, voxel) for c in clouds]
    pdist = [tree.query(c, k=1, workers=1)[0] for c in clouds]
    rng = np.random.default_rng(0)
    orders = {"spread": farthest_order(origins), "random": rng.permutation(T).tolist(), "sequential": list(range(T))}
    rows = []
    for oname, order in orders.items():
        occupied = np.zeros(0, np.int64)
        for k, v in enumerate(order):
            kv, pv, dv = keys[v], clouds[v], pdist[v]
            if len(kv) == 0:
                continue
            inocc = np.isin(kv, occupied, assume_unique=False) if len(occupied) else np.zeros(len(kv), bool)
            row = dict(scene=sc, seq=sq, n_steps=T, order=oname, k=k + 1, view=int(v), n_pts=int(len(kv)),
                       overlap_frac=float(inocc.mean()), prec_pts_all=float((dv < tau).mean()),
                       prec_overlap_pts=float((dv[inocc] < tau).mean()) if inocc.any() else float("nan"),
                       prec_new_pts=float((dv[~inocc] < tau).mean()) if (~inocc).any() else float("nan"))
            if k > 0:
                inc = origins[order[:k]]
                dd = np.linalg.norm(inc - origins[v], axis=1); j = int(np.argmin(dd))
                row["dist_nearest_m"] = float(dd[j])
                row["angle_nearest_deg"] = float(np.degrees(np.arctan2(dd[j], med_depth[v])))
            else:
                row["dist_nearest_m"] = float("nan"); row["angle_nearest_deg"] = float("nan")
            # new voxels: mean position of the view's points in voxels nobody occupied
            newk = kv[~inocc]
            if len(newk):
                uk, pos, cnt, inv = voxelize(pv[~inocc], voxel)
                e, _ = tree.query(pos, k=1, workers=1)
                row.update(n_new_vox=int(len(uk)), prec_new_vox=float((e < tau).mean()),
                           wrong_near=int(((e >= tau) & (e < 0.5)).sum()), wrong_disp=int(((e >= 0.5) & (e < 1.0)).sum()),
                           wrong_hall=int((e >= 1.0).sum()))
            else:
                row.update(n_new_vox=0, prec_new_vox=float("nan"), wrong_near=0, wrong_disp=0, wrong_hall=0)
            rows.append(row)
            occupied = np.union1d(occupied, kv)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--convention", default="right0")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "E101_marginal_view")
    a = ap.parse_args()
    a.pred_dir = a.pred_dir or str(REPO / "outputs" / "pred" / a.mode)
    CFG.update({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()})
    out = a.out / f"{a.mode}{a.tag}{'' if a.split == 'test' else '_val'}"; out.mkdir(parents=True, exist_ok=True)
    scenes = list(TEST_SCENES if a.split == "test" else VAL_SCENES)
    jobs = [(sc, sq) for sc in scenes for sq in sequences(sc)]
    t0 = time.time(); rows = []
    with Pool(a.workers, initializer=lambda c: CFG.update(c), initargs=(dict(CFG),)) as pool:
        for r in pool.imap_unordered(one_sequence, jobs):
            rows += r
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(out / "per_view.csv", index=False)
    (out / "config.yaml").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    (out / "git_commit.txt").write_text(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout)
    print(f"{len(df)} rows in {(time.time()-t0)/60:.1f} min -> {out}")

    # ---- summary against CRITERIA_E101
    from scipy.stats import spearmanr
    lines = [f"# E101 {a.mode}{a.tag} {a.split}: {df.groupby(["scene","seq"]).ngroups} sequences"]
    for oname in ("spread", "random", "sequential"):
        d = df[(df.order == oname) & (df.k >= 2)].dropna(subset=["prec_new_vox", "dist_nearest_m"])
        rho = spearmanr(d.dist_nearest_m, d.prec_new_vox).correlation
        bins = [(0, 0.5), (0.5, 1.5), (1.5, 99)]
        by = {f"{lo}-{hi}m": (float(d[(d.dist_nearest_m >= lo) & (d.dist_nearest_m < hi)].prec_new_vox.mean()), int(((d.dist_nearest_m >= lo) & (d.dist_nearest_m < hi)).sum())) for lo, hi in bins}
        lines.append(f"[{oname:10s}] H1 Spearman(dist to nearest included view, new-voxel precision) = {rho:+.3f} (n {len(d)}); new-voxel precision by distance: " +
                     ", ".join(f"{k}: {v[0]:.3f} (n {v[1]})" for k, v in by.items()))
        wn, wd, wh = d.wrong_near.sum(), d.wrong_disp.sum(), d.wrong_hall.sum(); tot = max(wn + wd + wh, 1)
        lines.append(f"[{oname:10s}] H2 wrong new voxels: near miss 0.2-0.5 m {wn/tot:.1%}, displaced 0.5-1 m {wd/tot:.1%}, hallucinated >1 m {wh/tot:.1%} (n {tot})")
        g = df[(df.order == oname) & (df.k >= 2)].groupby("k")[["prec_overlap_pts", "prec_new_pts", "overlap_frac", "prec_new_vox"]].mean()
        ks = [k for k in (2, 4, 8, 16, 24) if k in g.index]
        lines.append(f"[{oname:10s}] H3 precision overlap vs new points by k: " + "; ".join(f"k={k}: {g.loc[k,'prec_overlap_pts']:.3f} vs {g.loc[k,'prec_new_pts']:.3f} (overlap {g.loc[k,'overlap_frac']:.2f}, new-vox prec {g.loc[k,'prec_new_vox']:.3f})" for k in ks))
        gmin = (g.prec_overlap_pts - g.prec_new_pts).min()
        lines.append(f"[{oname:10s}] H3 min over k>=2 of (overlap - new) = {gmin:+.3f}")
    d = df[(df.order == "spread") & (df.k >= 2)].dropna(subset=["prec_new_vox"])
    d = d.assign(bin=np.where(d.dist_nearest_m < 0.5, "<0.5", np.where(d.dist_nearest_m < 1.5, "0.5-1.5", ">=1.5")))
    per = d.groupby(["scene", "bin"]).prec_new_vox.agg(["mean", "std", "count"]).round(3)
    lines.append("per scene, spread order, new-voxel precision by distance bin:\n" + per.to_string())
    txt = "\n".join(lines); print(txt); (out / "summary.txt").write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
