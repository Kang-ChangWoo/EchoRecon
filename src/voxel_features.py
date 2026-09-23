#!/usr/bin/env python3
"""E106: per-voxel aggregates of the per-observation "edge messages" a bipartite graph would
route, plus the label (voxel within 0.2 m of the fixed reference). One row per voxel of the
fused cloud at N in {4, all} (seed 0). Uses the posterior-head predictions (<mode>_post:
argmax depth + per-ray confidence). Criteria: results/E100_cause/CRITERIA_E104_E106.md.

    python src/voxel_features.py --mode r2 --split val
"""
from __future__ import annotations

import argparse
import json
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
from data import TEST_SCENES, VAL_SCENES, Sequence, sequences, scenes as all_scenes  # noqa: E402
from erp import quat_to_R, ray_dirs, to_radial  # noqa: E402
from eval_fusion import resize_nearest  # noqa: E402
from stage_a import subset  # noqa: E402

CFG = {}
FEATS = ["count", "views", "cells0.3", "cells0.6", "cells1.0", "span", "contra", "spread_deg", "conf_mean", "conf_max",
         "conf_min", "range_mean", "range_min", "abs_elev_mean", "N", "frac_views"]


def one(job):
    sc, sq = job; a = CFG; voxel, stride, md, margin = a["voxel"], a["stride"], a["max_depth"], 0.2
    pf = Path(a["pred_dir"]) / sc / f"{sq}.npz"
    z = np.load(pf); P = z["pred"].astype(np.float32); C = z["conf"].astype(np.float32); steps = z["steps"].tolist()
    S = Sequence(sc, sq); T = len(steps); H, W = P.shape[1:]
    dirs = ray_dirs(H, W, a["convention"]); dirs_s = dirs[::stride, ::stride]
    origins, Rs, clouds, confs, gts, elevs, free_sets, own_sets = [], [], [], [], [], [], [], []
    for k, i in enumerate(steps):
        pose = S.pose(i); R = quat_to_R(pose["rotation"]); o = np.asarray(pose["position"], float); origins.append(o); Rs.append(R)
        d = to_radial(P[k], dirs, "face")[::stride, ::stride]; v = np.isfinite(d) & (d > 0) & (d < md)
        pts = (dirs_s[v] * d[v][:, None]) @ R.T + o
        clouds.append(pts); confs.append(C[k][::stride, ::stride][v]); elevs.append(np.abs(dirs_s[v][:, 1]))
        g = to_radial(resize_nearest(S.gt_depth(i, "face"), (H, W)), dirs, "face")[::stride, ::stride]
        gv = np.isfinite(g) & (g > 0) & (g < md); gts.append((dirs_s[gv] * g[gv][:, None]) @ R.T + o)
        dd = dirs_s[v] @ R.T; dk = d[v] - margin; keys = []
        for j in range(int(np.ceil(max(dk.max(), 0) / voxel)) if dk.size else 0):
            t = (j + 0.5) * voxel; m = t < dk
            if not m.any():
                break
            keys.append(pack(o + dd[m] * t, voxel))
        free_sets.append(np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64))
        own_sets.append(np.unique(pack(pts, voxel)))
    origins = np.stack(origins); u = origins[-1] - origins[0]; u /= max(np.linalg.norm(u), 1e-9); proj = (origins - origins[0]) @ u
    _, ref, _, _ = voxelize(np.concatenate(gts), voxel / 2); tree = cKDTree(ref)
    out = []
    for N in (4, 0):
        idx = subset(T, N, 0)
        allP = np.concatenate([clouds[j] for j in idx]); view_of = np.concatenate([np.full(len(clouds[j]), j) for j in idx])
        cpts = np.concatenate([confs[j] for j in idx]); epts = np.concatenate([elevs[j] for j in idx])
        uk, pos, cnt, inv = voxelize(allP, voxel); nv = len(uk)
        pv = np.unique(inv.astype(np.int64) * 4096 + view_of); vox_of, vw = pv // 4096, pv % 4096
        F = {}
        F["count"] = cnt.astype(np.float32); F["views"] = np.bincount(vox_of, minlength=nv).astype(np.float32)
        for D in (0.3, 0.6, 1.0):
            pc = np.unique(vox_of * 4096 + np.floor(proj[vw] / D).astype(np.int64)); F[f"cells{D}"] = np.bincount(pc // 4096, minlength=nv).astype(np.float32)
        mx = np.full(nv, -np.inf); mn = np.full(nv, np.inf); np.maximum.at(mx, vox_of, proj[vw]); np.minimum.at(mn, vox_of, proj[vw]); F["span"] = (mx - mn).astype(np.float32)
        fs = np.concatenate([free_sets[j] for j in idx]); fu, fc = np.unique(fs, return_counts=True)
        c_free = np.zeros(nv, np.int64)
        if len(fu):
            h = np.minimum(np.searchsorted(fu, uk), len(fu) - 1); m_ = fu[h] == uk; c_free[m_] = fc[h[m_]]
        both = np.zeros(nv, np.int64)
        for j in idx:
            b_ = np.intersect1d(free_sets[j], own_sets[j], assume_unique=True)
            if len(b_):
                h = np.minimum(np.searchsorted(b_, uk), len(b_) - 1); both += (b_[h] == uk)
        F["contra"] = np.maximum(c_free - both, 0).astype(np.float32)
        dv = pos[inv] - origins[view_of]; rng_ = np.linalg.norm(dv, axis=1); dvn = dv / np.maximum(rng_[:, None], 1e-9)
        md_ = np.stack([np.bincount(inv, weights=dvn[:, d_], minlength=nv) for d_ in range(3)], 1) / cnt[:, None]
        F["spread_deg"] = np.degrees(np.arccos(np.clip(np.linalg.norm(md_, axis=1), -1, 1))).astype(np.float32)
        F["conf_mean"] = (np.bincount(inv, weights=cpts, minlength=nv) / cnt).astype(np.float32)
        cmx = np.full(nv, -np.inf); cmn = np.full(nv, np.inf); np.maximum.at(cmx, inv, cpts); np.minimum.at(cmn, inv, cpts); F["conf_max"] = cmx.astype(np.float32); F["conf_min"] = cmn.astype(np.float32)
        F["range_mean"] = (np.bincount(inv, weights=rng_, minlength=nv) / cnt).astype(np.float32)
        rmn = np.full(nv, np.inf); np.minimum.at(rmn, inv, rng_); F["range_min"] = rmn.astype(np.float32)
        F["abs_elev_mean"] = (np.bincount(inv, weights=epts, minlength=nv) / cnt).astype(np.float32)
        F["N"] = np.full(nv, len(idx), np.float32); F["frac_views"] = (F["views"] / len(idx)).astype(np.float32)
        err, _ = tree.query(pos, k=1, workers=1)
        X = np.stack([F[f] for f in FEATS], 1)
        out.append(dict(scene=sc, seq=sq, N_req=(N if N > 0 else -1), N=len(idx), X=X, y=(err < 0.2).astype(np.uint8), err=err.astype(np.float32), pos=pos.astype(np.float32)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="r2"); ap.add_argument("--split", default="val"); ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--voxel", type=float, default=0.1); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--convention", default="right0"); ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args(); a.pred_dir = a.pred_dir or str(REPO / "outputs" / "pred" / f"{a.mode}_post")
    CFG.update(vars(a))
    held = list(TEST_SCENES) + list(VAL_SCENES)
    scenes = {"test": list(TEST_SCENES), "val": list(VAL_SCENES), "train": [x for x in all_scenes() if x not in held]}[a.split]
    jobs = [(sc, sq) for sc in scenes for sq in sequences(sc)]
    out = REPO / "results" / "E106_learned" / "features"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); allrows = []
    with Pool(a.workers, initializer=lambda c: CFG.update(c), initargs=(dict(CFG),)) as pool:
        for r in pool.imap_unordered(one, jobs):
            allrows += r
    np.savez_compressed(out / f"{a.mode}_{a.split}.npz", rows=np.array(allrows, dtype=object), feats=np.array(FEATS), allow_pickle=True)
    n = sum(len(r["y"]) for r in allrows); pos = sum(int(r["y"].sum()) for r in allrows)
    print(f"{a.mode} {a.split}: {len(allrows)} (seq,N) rows, {n} voxels, positive rate {pos/n:.3f}, {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
