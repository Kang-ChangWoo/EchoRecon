#!/usr/bin/env python3
"""E110: top-k candidates per ray, between point fusion (k = 1) and the full distribution.
From <mode>_post_topk (k highest-mass local modes per ray with their 3-bin mass): candidate
points = the top-k mode depths along each ray; rankings on the union voxels
  topk_count<k>   number of rays whose top-k contains the voxel
  topk_mass<k>    summed mode mass
B_argmax (the head's argmax point, count ranking) is the point-fusion control on its own voxels.
Criteria: results/E100_cause/CRITERIA_E108_E110.md.

    python src/topk_fusion.py --mode r2 --split test
"""
from __future__ import annotations

import argparse
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
from cause_decomp import voxelize  # noqa: E402
from conf_loss_decomp import normals  # noqa: E402
from data import TEST_SCENES, VAL_SCENES, Sequence, sequences  # noqa: E402
from erp import face_cos, quat_to_R, ray_dirs  # noqa: E402
from eval_fusion import resize_nearest  # noqa: E402
from fuse import metrics  # noqa: E402
from stage_a import subset  # noqa: E402

NS = (1, 2, 4, 8, 16, 0)
FRACS = (0.5, 0.25, 0.1)
KS = (1, 2, 3, 5)
CFG = {}


def one(job):
    sc, sq = job; a = CFG; voxel, stride, md = a["voxel"], a["stride"], a["max_depth"]
    z = np.load(Path(a["pred_dir"]) / sc / f"{sq}.npz"); P = z["pred"].astype(np.float32); steps = z["steps"].tolist()
    TI = z["topk_idx"].astype(np.int64); TM = z["topk_mass"].astype(np.float32); centres = z["centres"].astype(np.float32)
    S = Sequence(sc, sq); T = len(steps); H, W = P.shape[1:]; dirs = ray_dirs(H, W, a["convention"]); ds = dirs[::stride, ::stride]; fc = face_cos(ds)
    origins, arg_pts, gts, cand = [], [], [], []      # cand[j] = (points (n,kmax,3), mass (n,kmax))
    for k, i in enumerate(steps):
        pose = S.pose(i); R = quat_to_R(pose["rotation"]); o = np.asarray(pose["position"], float); origins.append(o)
        d = P[k][::stride, ::stride] / fc; v = np.isfinite(d) & (d > 0) & (d < md); arg_pts.append((ds[v] * d[v][:, None]) @ R.T + o)
        dk = centres[TI[k][:, ::stride, ::stride]] / fc[None]          # (kmax, h, w) radial depth of each mode
        mk = TM[k][:, ::stride, ::stride]
        okk = (dk > 0) & (dk < md) & (mk > 0)
        pts = (ds[None] * dk[..., None]) @ R.T + o                      # (kmax, h, w, 3)
        cand.append((pts, mk, okk))
        g = resize_nearest(S.gt_depth(i, "face"), (H, W))[::stride, ::stride] / fc; gv = np.isfinite(g) & (g > 0) & (g < md); gts.append((ds[gv] * g[gv][:, None]) @ R.T + o)
    origins = np.stack(origins); ref_fixed = voxelize(np.concatenate(gts), voxel / 2)[1]; nrm, _ = normals(ref_fixed)
    r_ref = np.linalg.norm(ref_fixed[:, None, :] - origins[None, :, :], axis=2); jn = r_ref.argmin(1); r_ref = r_ref[np.arange(len(ref_fixed)), jn]
    vdir = ref_fixed - origins[jn]; vdir /= np.maximum(np.linalg.norm(vdir, axis=1, keepdims=True), 1e-9)
    ref_bin = np.digitize(r_ref, [2, 4]) * 3 + np.digitize(np.degrees(np.arccos(np.clip(np.abs((vdir * nrm).sum(1)), 0, 1))), [30, 60]); nref_bin = np.bincount(ref_bin, minlength=9)
    rows = []
    for N in NS:
        idx = subset(T, N, 0)
        refs = {"fixed": ref_fixed, "subset": voxelize(np.concatenate([gts[j] for j in idx]), voxel / 2)[1]}
        base = dict(scene=sc, seq=sq, n_steps=T, N_req=(N if N > 0 else -1), N=len(idx))

        def add(name, pts, sc_, extra):
            order = np.argsort(-sc_, kind="stable")
            for fr in FRACS:
                keep = order[: max(1, int(fr * len(pts)))]
                for rn, ref in refs.items():
                    m = metrics(pts[keep], ref, voxel=voxel)
                    if rn == "fixed":
                        dk_, _ = cKDTree(pts[keep]).query(ref_fixed, k=1, workers=1); hit = np.bincount(ref_bin, weights=dk_ < 0.2, minlength=9)
                        m.update({f"hit_b{b}": float(hit[b]) for b in range(9)}, **{f"nref_b{b}": int(nref_bin[b]) for b in range(9)})
                    rows.append({**base, "ranking": name, "ref": rn, "frac": fr, "n_cand": len(pts), **extra, **m})

        allA = np.concatenate([arg_pts[j] for j in idx]); _, posA, cntA, _ = voxelize(allA, voxel); add("B_argmax", posA, cntA.astype(float), {"k": 1})
        for kk in KS:
            pts = np.concatenate([cand[j][0][:kk][cand[j][2][:kk]] for j in idx]); mass = np.concatenate([cand[j][1][:kk][cand[j][2][:kk]] for j in idx])
            _, pos, cnt, inv = voxelize(pts, voxel)
            add(f"topk_count{kk}", pos, cnt.astype(float), {"k": kk}); add(f"topk_mass{kk}", pos, np.bincount(inv, weights=mass, minlength=len(pos)), {"k": kk})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="r2"); ap.add_argument("--split", default="test"); ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--voxel", type=float, default=0.1); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--convention", default="right0"); ap.add_argument("--workers", type=int, default=20); ap.add_argument("--seq-limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "E110_topk")
    a = ap.parse_args(); a.pred_dir = a.pred_dir or str(REPO / "outputs" / "pred" / f"{a.mode}_post_topk"); CFG.update({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()})
    scenes = TEST_SCENES if a.split == "test" else VAL_SCENES
    jobs = [(sc, sq) for sc in scenes for sq in (sequences(sc)[: a.seq_limit] if a.seq_limit else sequences(sc))]
    out = a.out / f"{a.mode}{'' if a.split == 'test' else '_val'}"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); rows = []
    with Pool(a.workers, initializer=lambda c: CFG.update(c), initargs=(dict(CFG),)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs)):
            rows += r; print(f"[{i+1}/{len(jobs)}] {len(rows)} rows {(time.time()-t0)/60:.1f} min", flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "per_sequence.csv", index=False)
    (out / "git_commit.txt").write_text(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout)
    print(f"{len(rows)} rows in {(time.time()-t0)/60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
