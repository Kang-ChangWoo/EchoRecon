#!/usr/bin/env python3
"""Stage A (E00-E07): does the reconstruction really get worse with more views?

One pass per sequence: unproject every step once, then evaluate every
(number of views N, subset seed) combination on that material, so the view-count
curve costs one unprojection rather than one per point of the curve.

Per (N, seed) it reports the full metric set (accuracy, completeness, symmetric
Chamfer, precision / recall / F1 at three thresholds, voxel IoU, point count)
for the fused cloud and for the support-ranked and oracle-ranked top fractions,
which gives the accuracy-completeness Pareto curve (E05, E06) and the oracle
headroom (E07) at the same time.

Subsets: seed 0 keeps the deterministic evenly-spaced subset; seeds 1+ draw N
steps at random without replacement, so the curve carries a spread over subset
choice rather than one arbitrary choice.

    python src/stage_a.py --mode r2 --out results/E01_view_curve
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from data import TEST_SCENES, Sequence, sequences  # noqa: E402
from erp import ray_dirs  # noqa: E402
from eval_fusion import resize_nearest, unproject_res  # noqa: E402
from fuse import metrics, voxel_downsample  # noqa: E402

NS = (1, 2, 4, 8, 16, 32, 0)          # 0 = all steps
FRACS = (1.0, 0.9, 0.75, 0.5, 0.25, 0.1)
CFG = {}


def subset(n_steps: int, N: int, seed: int) -> np.ndarray:
    if N <= 0 or N >= n_steps:
        return np.arange(n_steps)
    if seed == 0:
        return np.linspace(0, n_steps - 1, N).astype(int)
    return np.sort(np.random.default_rng(seed * 1000 + N).choice(n_steps, N, replace=False))


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
    clouds, refs = [], []
    for k, i in enumerate(steps):
        clouds.append(unproject_res(P[k], S.pose(i), dirs, a["max_depth"], a["stride"], "face")[0])
        g = resize_nearest(S.gt_depth(i, "face"), (H, W))
        refs.append(unproject_res(g, S.pose(i), dirs, a["max_depth"], a["stride"], "face")[0])
    rows = []
    for N in NS:
        for seed in range(a["seeds"]):
            if N <= 0 and seed > 0:
                continue                      # "all" has only one subset
            idx = subset(len(steps), N, seed)
            if len(idx) < 1:
                continue
            allP = np.concatenate([clouds[j] for j in idx])
            ref, _ = voxel_downsample(np.concatenate([refs[j] for j in idx]), a["voxel"] / 2)
            vox, count = voxel_downsample(allP, a["voxel"])
            from scipy.spatial import cKDTree
            err, _ = cKDTree(ref).query(vox, k=1, workers=-1)
            # N_req is the requested number of views (-1 = every step); N is what the
            # sequence could actually give, which is smaller for short sequences
            base = dict(scene=sc, seq=sq, n_steps=len(steps), N_req=(N if N > 0 else -1),
                        N=len(idx), seed=seed)
            for rank, sc_ in (("support", count), ("oracle", -err)):
                order = np.argsort(-sc_)
                for fr in FRACS:
                    keep = order[: max(1, int(fr * len(order)))]
                    m = metrics(vox[keep], ref, voxel=a["voxel"])
                    rows.append({**base, "ranking": rank, "frac": fr, **m})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--scenes", nargs="+", default=list(TEST_SCENES))
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--convention", default="right0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "E01_view_curve")
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
    keys = ["ranking", "N_req", "frac"]
    num = df.select_dtypes("number").columns.tolist()
    df.groupby(keys)[num].agg(["mean", "std"]).to_csv(out / "aggregate.csv")
    df.groupby(["scene"] + keys)[num].mean().to_csv(out / "per_scene.csv")
    df.groupby(["seed"] + keys)[num].mean().to_csv(out / "per_seed.csv")
    (out / "config.yaml").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    (out / "git_commit.txt").write_text(
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout)
    summary = {}
    for N in sorted(df.N_req.unique()):
        d = df[(df.N_req == N) & (df.ranking == "support") & (df.frac == 1.0)]
        o = df[(df.N_req == N) & (df.ranking == "oracle") & (df.frac == 0.25)]
        s = df[(df.N_req == N) & (df.ranking == "support") & (df.frac == 0.25)]
        summary[int(N)] = {
            "n_rows": int(len(d)), "n_views_mean": float(d.N.mean()),
            "all": {k: float(d[k].mean()) for k in ("acc", "comp", "chamfer", "f1@0.2", "iou", "n_pred")},
            "all_std_acc": float(d.groupby("seed")["acc"].mean().std()) if d.seed.nunique() > 1 else 0.0,
            "support_top25": {k: float(s[k].mean()) for k in ("acc", "comp", "chamfer", "f1@0.2", "iou")},
            "oracle_top25": {k: float(o[k].mean()) for k in ("acc", "comp", "chamfer", "f1@0.2", "iou")}}
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{a.mode}: {len(df)} rows in {time.time()-t0:.0f} s -> {out}")
    print(f"{'N':>4s} {'used':>5s} {'acc':>7s} {'comp':>7s} {'chamfer':>8s} {'F1@0.2':>7s} {'IoU':>6s} {'points':>8s} "
          f"| {'sup25 acc':>9s} {'sup25 F1':>8s} | {'orc25 acc':>9s} {'orc25 F1':>8s}")
    for N, v in summary.items():
        A, S_, O = v["all"], v["support_top25"], v["oracle_top25"]
        print(f"{N:4d} {v['n_views_mean']:5.1f} {A['acc']:7.3f} {A['comp']:7.3f} {A['chamfer']:8.3f} {A['f1@0.2']:7.3f} {A['iou']:6.3f} "
              f"{A['n_pred']:8.0f} | {S_['acc']:9.3f} {S_['f1@0.2']:8.3f} | {O['acc']:9.3f} {O['f1@0.2']:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
