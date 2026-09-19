#!/usr/bin/env python3
"""Stage B (E10-E15): how far does point-only robust fusion get, without any posterior?

STOP CHECK A. If a strong point-only fusion closes most of the gap between
counting support and the oracle, then the posterior hypothesis is not needed and
the direction changes. This builds that baseline honestly and sweeps each
method's parameter so the comparison is a curve, not one operating point.

All methods act on the same fused voxel cloud (positions are the plain average
of the points in each voxel), so they differ only in which voxels they keep:

  union             every voxel
  support_frac      top fraction by contributing-point count
  support_min       at least k contributing points
  sor               statistical outlier removal: drop voxels whose mean distance
                    to their k nearest voxels exceeds mean + z * std
  ror               radius outlier removal: drop voxels with fewer than n
                    neighbours within r
  cluster           26-connected components of the occupied grid, keep
                    components of at least m voxels
  consensus         keep voxels whose contributing points agree: the spread
                    (root mean square deviation from the voxel mean) is below s
  keyframe          union, but only over views at least delta apart
  tsdf              truncated signed distance surface, weight threshold w

Reported per method and parameter: the full metric set. The headline per method
is the best F1@0.2 over its sweep, with the accuracy, completeness and kept
fraction at that point, against the oracle ranking's curve.

    python src/stage_b.py --mode r2 --out results/E10_robust_point
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
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from data import TEST_SCENES, Sequence, sequences  # noqa: E402
from erp import ray_dirs  # noqa: E402
from eval_fusion import resize_nearest, unproject_res  # noqa: E402
from fuse import TSDF, metrics, voxel_downsample  # noqa: E402
from stage_a import subset  # noqa: E402

CFG = {}


def connected_keep(vox, voxel, min_size):
    """Indices of voxels in a 26-connected component of at least min_size voxels."""
    key = np.floor(vox / voxel).astype(np.int64)
    lo = key.min(0)
    idx = key - lo
    shape = idx.max(0) + 1
    if np.prod(shape) > 4e8:
        return np.arange(len(vox))
    grid = np.zeros(shape, bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())
    mine = lab[idx[:, 0], idx[:, 1], idx[:, 2]]
    return np.nonzero(sizes[mine] >= min_size)[0]


def one_sequence(args):
    sc, sq = args
    a = CFG
    f = REPO / "outputs" / "pred" / a["mode"] / sc / f"{sq}.npz"
    if not f.exists():
        return []
    z = np.load(f)
    P = z["pred"].astype(np.float32); steps = z["steps"].tolist()
    S = Sequence(sc, sq)
    H, W = P.shape[1:]
    dirs = ray_dirs(H, W, a["convention"])
    clouds, refs, origins, dirw, drange = [], [], [], [], []
    for k, i in enumerate(steps):
        pts, valid, d, dw = unproject_res(P[k], S.pose(i), dirs, a["max_depth"], a["stride"], "face")
        clouds.append(pts); origins.append(np.asarray(S.pose(i)["position"], float))
        dirw.append(dw); drange.append(d)
        g = resize_nearest(S.gt_depth(i, "face"), (H, W))
        refs.append(unproject_res(g, S.pose(i), dirs, a["max_depth"], a["stride"], "face")[0])
    rows = []
    for N in a["ns"]:
        idx = subset(len(steps), N, 0)
        allP = np.concatenate([clouds[j] for j in idx])
        ref, _ = voxel_downsample(np.concatenate([refs[j] for j in idx]), a["voxel"] / 2)
        vox, count = voxel_downsample(allP, a["voxel"])
        # per-voxel spread of the contributing points, for the consensus filter
        key = np.floor(allP / a["voxel"]).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        sq_sum = np.zeros(len(vox))
        for c in range(3):
            m = np.bincount(inv, weights=allP[:, c], minlength=len(vox)) / np.maximum(count, 1)
            sq_sum += np.bincount(inv, weights=(allP[:, c] - m[inv]) ** 2, minlength=len(vox))
        spread = np.sqrt(sq_sum / np.maximum(count, 1))
        err, _ = cKDTree(ref).query(vox, k=1, workers=-1)
        tree = cKDTree(vox)
        base = dict(scene=sc, seq=sq, N_req=(N if N > 0 else -1), N=len(idx), n_vox=len(vox))

        def add(method, param, keep):
            keep = np.asarray(keep)
            if len(keep) == 0:
                return
            m = metrics(vox[keep], ref, voxel=a["voxel"])
            rows.append({**base, "method": method, "param": param,
                         "kept_frac": len(keep) / len(vox), **m})

        add("union", 0, np.arange(len(vox)))
        order = np.argsort(-count)
        for fr in (0.9, 0.75, 0.5, 0.25, 0.1):
            add("support_frac", fr, order[: max(1, int(fr * len(vox)))])
        for k in (2, 3, 4, 5, 8, 12):
            add("support_min", k, np.nonzero(count >= k)[0])
        dk, _ = tree.query(vox, k=min(21, len(vox)), workers=-1)
        mdist = dk[:, 1:].mean(1)
        for zt in (2.0, 1.0, 0.5, 0.0):
            add("sor", zt, np.nonzero(mdist <= mdist.mean() + zt * mdist.std())[0])
        for r in (0.15, 0.3):
            cnt_r = np.array(tree.query_ball_point(vox, r, workers=-1, return_length=True))
            for mn in (4, 8, 16, 32):
                add("ror", f"{r}/{mn}", np.nonzero(cnt_r >= mn)[0])
        for ms in (5, 20, 100, 500, 2000):
            add("cluster", ms, connected_keep(vox, a["voxel"], ms))
        for sp in (0.06, 0.05, 0.04, 0.03):
            add("consensus", sp, np.nonzero(spread <= sp)[0])
        for delta in (0.3, 0.6, 1.0):
            keepv = [0]
            for j in range(1, len(idx)):
                if np.linalg.norm(origins[idx[j]] - origins[idx[keepv[-1]]]) >= delta:
                    keepv.append(j)
            sub = np.concatenate([clouds[idx[j]] for j in keepv])
            v2, _ = voxel_downsample(sub, a["voxel"])
            m = metrics(v2, ref, voxel=a["voxel"])
            rows.append({**base, "method": "keyframe", "param": delta,
                         "kept_frac": len(v2) / len(vox), **m})
        if a["tsdf"]:
            lo = allP.min(0) - 0.5; hi = allP.max(0) + 0.5
            t = TSDF(lo, hi, a["voxel"], a["trunc"])
            for j in idx:
                t.integrate(origins[j], dirw[j], drange[j], np.ones_like(drange[j], np.float32))
            for w in (1.0, 5.0, 20.0, 50.0):
                surf, _ = t.surface_points(min_w=w)
                if len(surf):
                    m = metrics(surf, ref, voxel=a["voxel"])
                    rows.append({**base, "method": "tsdf", "param": w,
                                 "kept_frac": len(surf) / len(vox), **m})
        oo = np.argsort(err)
        for fr in (0.9, 0.75, 0.5, 0.25, 0.1):
            add("oracle_frac", fr, oo[: max(1, int(fr * len(vox)))])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--scenes", nargs="+", default=list(TEST_SCENES))
    ap.add_argument("--ns", type=int, nargs="+", default=[0, 4], help="view counts (0 = all)")
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--trunc", type=float, default=0.3)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--convention", default="right0")
    ap.add_argument("--tsdf", action="store_true", default=True)
    ap.add_argument("--no-tsdf", dest="tsdf", action="store_false")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "E10_robust_point")
    a = ap.parse_args()
    CFG.update(vars(a)); CFG["out"] = str(a.out)
    out = a.out / a.mode
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(sc, sq) for sc in a.scenes for sq in sequences(sc)]
    t0 = time.time()
    rows = []
    with Pool(a.workers, initializer=lambda c: CFG.update(c), initargs=(dict(CFG),)) as pool:
        for i, r in enumerate(pool.imap_unordered(one_sequence, jobs)):
            rows += r
            print(f"[{i+1}/{len(jobs)}] {len(rows)} rows", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_sequence.csv", index=False)
    keys = ["N_req", "method", "param"]
    num = [c for c in df.select_dtypes("number").columns if c not in keys]
    agg = df.groupby(keys)[num].mean().reset_index()
    agg.to_csv(out / "aggregate.csv", index=False)
    (out / "config.yaml").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    (out / "git_commit.txt").write_text(
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout)
    best = {}
    for N in sorted(agg.N_req.unique()):
        sub = agg[agg.N_req == N]
        tab = []
        for meth in sub.method.unique():
            m = sub[sub.method == meth]
            b = m.loc[m["f1@0.2"].idxmax()]
            tab.append((meth, b["param"], b["f1@0.2"], b["acc"], b["comp"], b["chamfer"], b["iou"], b["kept_frac"]))
        tab.sort(key=lambda r: -r[2])
        best[int(N)] = [dict(method=t[0], param=str(t[1]), f1=float(t[2]), acc=float(t[3]), comp=float(t[4]),
                             chamfer=float(t[5]), iou=float(t[6]), kept=float(t[7])) for t in tab]
        print(f"\nN = {'all' if N < 0 else N}: best F1@0.2 per method")
        print(f"{'method':14s} {'param':>8s} {'F1@0.2':>7s} {'acc':>7s} {'comp':>7s} {'chamfer':>8s} {'IoU':>6s} {'kept':>6s}")
        for t in tab:
            print(f"{t[0]:14s} {str(t[1]):>8s} {t[2]:7.3f} {t[3]:7.3f} {t[4]:7.3f} {t[5]:8.3f} {t[6]:6.3f} {t[7]:6.2f}")
    (out / "metrics.json").write_text(json.dumps(best, indent=1))
    print(f"\n{len(df)} rows in {time.time()-t0:.0f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
