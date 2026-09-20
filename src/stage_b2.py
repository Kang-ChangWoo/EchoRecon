#!/usr/bin/env python3
"""Stage B, second half (E25 control): does soft posterior fusion help by itself?

STOP CHECK B, run before any posterior is trained. The existing point
predictions are turned into posteriors of fixed width, p(d) proportional to
exp(-(d - d_hat)^2 / 2 sigma^2), and fused softly over a candidate voxel grid
that covers the room, not only the voxels some view's argmax happened to hit.
If a fake posterior already recovers much of the oracle gap, the gain belongs to
soft accumulation and not to learned ambiguity, and a learned posterior has to
beat this line to claim anything.

Rankings compared on the same sequences: cross-view support of the point cloud
(Stage A/B's best realisable ranking), the fake posterior's surface evidence at
each sigma, and the oracle.

    python src/stage_b2.py --mode r2 --out results/E25_fake_posterior
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
from data import TEST_SCENES, Sequence, sequences  # noqa: E402
from erp import ray_dirs  # noqa: E402
from eval_fusion import resize_nearest, unproject_res  # noqa: E402
from fuse import metrics, voxel_downsample  # noqa: E402
from posterior import candidate_grid, depth_bins, fuse, view_from_prediction  # noqa: E402
from stage_a import subset  # noqa: E402

SIGMAS = (0.05, 0.10, 0.20, 0.40)
FRACS = (0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002)
CFG = {}


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
    edges, _ = depth_bins(a["bins"], 0.1, a["max_depth"])
    rows = []
    for N in a["ns"]:
        idx = subset(len(steps), N, 0)
        clouds, refs = [], []
        for j in idx:
            clouds.append(unproject_res(P[j], S.pose(steps[j]), dirs, a["max_depth"], a["stride"], "face")[0])
            g = resize_nearest(S.gt_depth(steps[j], "face"), (H, W))
            refs.append(unproject_res(g, S.pose(steps[j]), dirs, a["max_depth"], a["stride"], "face")[0])
        ref, _ = voxel_downsample(np.concatenate(refs), a["voxel"] / 2)
        allP = np.concatenate(clouds)
        vox, count = voxel_downsample(allP, a["voxel"])
        base = dict(scene=sc, seq=sq, N_req=(N if N > 0 else -1), N=len(idx), n_vox=len(vox))
        err, _ = cKDTree(ref).query(vox, k=1, workers=-1)

        def add(method, param, pts, keep, n_cand):
            m = metrics(pts[keep], ref, voxel=a["voxel"])
            rows.append({**base, "method": method, "param": param, "n_cand": n_cand,
                         "kept": len(keep), "kept_frac": len(keep) / n_cand, **m})

        # the point pipeline's own ranking, for reference on the same sequence
        order = np.argsort(-count)
        for fr in FRACS:
            add("support", fr, vox, order[: max(1, int(fr * len(vox)))], len(vox))
        oo = np.argsort(err)
        for fr in FRACS:
            add("oracle", fr, vox, oo[: max(1, int(fr * len(vox)))], len(vox))
        # candidate grid over the room, so posterior fusion may support voxels no
        # argmax ever hit; capped so a long trajectory cannot blow up memory
        lo = np.minimum(ref.min(0), allP.min(0)) - 0.2
        hi = np.maximum(ref.max(0), allP.max(0)) + 0.2
        n_est = np.prod((hi - lo) / a["voxel"] + 1)
        if n_est > a["max_cand"]:
            scale = (n_est / a["max_cand"]) ** (1 / 3)
            cand = candidate_grid(lo, hi, a["voxel"] * scale)
        else:
            cand = candidate_grid(lo, hi, a["voxel"])
        for sigma in SIGMAS:
            views = [view_from_prediction(P[j], S.pose(steps[j]), edges, sigma, a["convention"]) for j in idx]
            # (a) over the whole room grid: the posterior may support voxels no argmax hit
            A = fuse(views, cand, tau=a["tau_band"])
            o2 = np.argsort(-A)
            for fr in FRACS:
                add("fake_posterior", sigma, cand, o2[: max(1, int(fr * len(cand)))], len(cand))
            # (b) on exactly the voxels the support ranking sees, so soft against hard
            # evidence is compared without the candidate set differing
            Av = fuse(views, vox, tau=a["tau_band"])
            o3 = np.argsort(-Av)
            for fr in FRACS:
                add("fake_posterior_same_cand", sigma, vox, o3[: max(1, int(fr * len(vox)))], len(vox))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--scenes", nargs="+", default=list(TEST_SCENES))
    ap.add_argument("--ns", type=int, nargs="+", default=[0, 4])
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--bins", type=int, default=128)
    ap.add_argument("--tau-band", type=float, default=0.15, help="half width of the surface band, metres")
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--max-cand", type=float, default=1.5e6)
    ap.add_argument("--convention", default="right0")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "E25_fake_posterior")
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
    # the kept fraction is part of the operating point: averaging over it before
    # taking the best per method (as an earlier version did) mixed eight
    # fractions into one number and produced the withdrawn -55 % line
    # `param` is the requested fraction for support/oracle but sigma for the fake
    # rows, whose fraction is only in the achieved `kept_frac`; snap it back to
    # the requested value so every method is summarised per operating point
    fr = np.array(sorted(FRACS))
    df["frac_req"] = fr[np.abs(np.log(df["kept_frac"].clip(1e-6).values[:, None]) - np.log(fr)[None, :]).argmin(1)]
    keys = ["N_req", "method", "param", "frac_req"]
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
            tab.append((meth, b["param"], b["f1@0.2"], b["acc"], b["comp"], b["chamfer"], b["iou"], b["kept"]))
        tab.sort(key=lambda r: -r[2])
        best[int(N)] = [dict(method=t[0], param=float(t[1]), f1=float(t[2]), acc=float(t[3]), comp=float(t[4]),
                             chamfer=float(t[5]), iou=float(t[6]), kept=float(t[7])) for t in tab]
        print(f"\nN = {'all' if N < 0 else N}: best F1@0.2 per method")
        print(f"{'method':16s} {'param':>7s} {'F1@0.2':>7s} {'acc':>7s} {'comp':>7s} {'chamfer':>8s} {'IoU':>6s} {'kept':>9s}")
        for t in tab:
            print(f"{t[0]:16s} {t[1]:7.2f} {t[2]:7.3f} {t[3]:7.3f} {t[4]:7.3f} {t[5]:8.3f} {t[6]:6.3f} {t[7]:9.0f}")
        d = {t[0]: t[2] for t in tab}
        if {"support", "oracle", "fake_posterior"} <= set(d):
            print(f"  fake posterior closes {(d['fake_posterior']-d['support'])/(d['oracle']-d['support'])*100:+.0f} % "
                  f"of the gap between the support ranking and the oracle")
    (out / "metrics.json").write_text(json.dumps(best, indent=1))
    print(f"\n{len(df)} rows in {time.time()-t0:.0f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
