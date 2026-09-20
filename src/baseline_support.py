#!/usr/bin/env python3
"""E102: baseline-aware support. Same voxels as Stage A, different rankings:
point count (control), distinct views, Δ-cell count along the trajectory,
span of the contributing views. Criteria: results/E100_cause/CRITERIA_E102.md.

    python src/baseline_support.py --mode r2 --split test
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from cause_decomp import voxelize  # noqa: E402
from data import TEST_SCENES, VAL_SCENES, Sequence, sequences  # noqa: E402
from erp import quat_to_R, ray_dirs, to_radial  # noqa: E402
from eval_fusion import resize_nearest  # noqa: E402
from fuse import metrics  # noqa: E402
from stage_a import subset  # noqa: E402

NS = (1, 2, 4, 8, 16, 0)
FRACS = (1.0, 0.5, 0.25, 0.1)
DELTAS = (0.3, 0.6, 1.0, 1.5)
CFG = {}


def one_sequence(job):
    sc, sq = job
    a = CFG; voxel, stride, md = a["voxel"], a["stride"], a["max_depth"]
    pf = Path(a["pred_dir"]) / sc / f"{sq}.npz"
    if not pf.exists():
        return []
    z = np.load(pf); P_face = z["pred"].astype(np.float32); steps = z["steps"].tolist()
    S = Sequence(sc, sq); T = len(steps); H, W = P_face.shape[1:]
    dirs = ray_dirs(H, W, a["convention"]); dirs_s = dirs[::stride, ::stride]
    origins, clouds, gts = [], [], []
    for k, i in enumerate(steps):
        pose = S.pose(i); R = quat_to_R(pose["rotation"]); o = np.asarray(pose["position"], float); origins.append(o)
        d = to_radial(P_face[k], dirs, "face")[::stride, ::stride]; v = np.isfinite(d) & (d > 0) & (d < md)
        clouds.append((dirs_s[v] * d[v][:, None]) @ R.T + o)
        g = to_radial(resize_nearest(S.gt_depth(i, "face"), (H, W)), dirs, "face")[::stride, ::stride]
        gv = np.isfinite(g) & (g > 0) & (g < md)
        gts.append((dirs_s[gv] * g[gv][:, None]) @ R.T + o)
    origins = np.stack(origins)
    u = origins[-1] - origins[0]; u = u / max(np.linalg.norm(u), 1e-9)
    proj = (origins - origins[0]) @ u                       # position along the trajectory (m)
    _, ref_fixed, _, _ = voxelize(np.concatenate(gts), voxel / 2)
    rows = []
    for N in NS:
        for seed in range(a["seeds"]):
            if N <= 0 and seed > 0:
                continue
            idx = subset(T, N, seed)
            _, ref_sub, _, _ = voxelize(np.concatenate([gts[j] for j in idx]), voxel / 2)
            refs = {"fixed": ref_fixed, "subset": ref_sub}
            allP = np.concatenate([clouds[j] for j in idx])
            view_of_pt = np.concatenate([np.full(len(clouds[j]), j) for j in idx])
            uk, pos, cnt, inv = voxelize(allP, voxel)
            nv = len(uk)
            pv = np.unique(inv.astype(np.int64) * 4096 + view_of_pt)          # (voxel, view) pairs
            vox_of, view_of = pv // 4096, pv % 4096
            scores = {"support": cnt.astype(np.float64)}
            scores["views"] = np.bincount(vox_of, minlength=nv) + 1e-3 * cnt
            for D in DELTAS:
                cell = np.floor(proj[view_of] / D).astype(np.int64)
                pc = np.unique(vox_of * 4096 + cell)
                scores[f"cells{D}"] = np.bincount(pc // 4096, minlength=nv) + 1e-3 * cnt
            mx = np.full(nv, -np.inf); mn = np.full(nv, np.inf)
            np.maximum.at(mx, vox_of, proj[view_of]); np.minimum.at(mn, vox_of, proj[view_of])
            scores["span"] = (mx - mn) + 1e-3 * cnt
            base = dict(scene=sc, seq=sq, n_steps=T, N_req=(N if N > 0 else -1), N=len(idx), seed=seed, n_vox=nv)
            for name, sc_ in scores.items():
                order = np.argsort(-sc_, kind="stable")
                for fr in FRACS:
                    if fr == 1.0 and name != "support":
                        continue                            # every ranking keeps the same cloud at f = 1
                    keep = order[: max(1, int(fr * nv))]
                    for rn, ref in refs.items():
                        m = metrics(pos[keep], ref, voxel=voxel)
                        rows.append({**base, "ranking": name, "frac": fr, "ref": rn, **m})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--convention", default="right0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "E102_baseline_support")
    a = ap.parse_args()
    a.pred_dir = a.pred_dir or str(REPO / "outputs" / "pred" / a.mode)
    CFG.update({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()})
    out = a.out / f"{a.mode}{a.tag}{'' if a.split == 'test' else '_val'}"; out.mkdir(parents=True, exist_ok=True)
    scenes = list(TEST_SCENES if a.split == "test" else VAL_SCENES)
    jobs = [(sc, sq) for sc in scenes for sq in sequences(sc)]
    t0 = time.time(); rows = []
    with Pool(a.workers, initializer=lambda c: CFG.update(c), initargs=(dict(CFG),)) as pool:
        for i, r in enumerate(pool.imap_unordered(one_sequence, jobs)):
            rows += r; print(f"[{i+1}/{len(jobs)}] {len(rows)} rows {(time.time()-t0)/60:.1f} min", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(out / "per_sequence.csv", index=False)
    (out / "config.yaml").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    (out / "git_commit.txt").write_text(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout)
    print(f"{len(df)} rows in {(time.time()-t0)/60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
