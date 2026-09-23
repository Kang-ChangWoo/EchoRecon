#!/usr/bin/env python3
"""E108 (expected-count normalisation) and E109 (continuity-aware accumulation) on the A_point
pipeline. Criteria: results/E100_cause/CRITERIA_E108_E110.md.

Rankings on the same voxels as the raw-count baseline:
  count                      baseline (Stage A)
  rate_geo_pred<Emin>        count / expected rays, expected from pose + predicted-surface visibility   (E108 A)
  rate_geo_gt<Emin>          same with GT-surface visibility (diagnostic upper bound only)               (E108 A')
  rate_smooth<sigma>         count / Gaussian-smoothed point density                                     (E108 B)
  cont_a                     baseline-weighted votes, weight 1 - rho(d to the previous voter)            (E109 a)
  cont_b<w>                  votes confirmed by a trajectory neighbour (within w steps of the subset)    (E109 b)
  cont_c<a>_<b>              recursive log-odds, +a vote / -b free-space traversal, (1 - rho) discounted (E109 c)
Thresholded E108 variants (rate >= t) are scored beside the count ranking at the same kept count.
Recall of the fixed reference by range and incidence bin is recorded for every row.

    python src/continuity_fusion.py --mode r2 --split test
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
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from cause_decomp import pack, voxelize  # noqa: E402
from conf_loss_decomp import normals  # noqa: E402
from data import TEST_SCENES, VAL_SCENES, Sequence, sequences  # noqa: E402
from erp import quat_to_R, ray_dirs, to_radial  # noqa: E402
from eval_fusion import resize_nearest  # noqa: E402
from fuse import metrics  # noqa: E402
from stage_a import subset  # noqa: E402

NS = (1, 2, 4, 8, 16, 0)
FRACS = (1.0, 0.5, 0.25, 0.1)
EMIN = (0.5, 1.0, 2.0)
SIGMAS = (0.3, 0.5, 1.0)
THRESH = (0.1, 0.2, 0.3, 0.5)
CONT_C = ((1.0, 0.5), (1.0, 1.0), (1.0, 2.0))
LAMBDAS = (0.3, 0.6, 1.0, 1.5)
CFG = {}
# E100 (c): correlation of neighbouring views' errors by baseline; piecewise linear in d
RHO_D = np.array([0.0, 0.25, 1.0, 2.0, 3.0]); RHO_V = np.array([0.95, 0.88, 0.73, 0.41, 0.30])


def rho(d):
    return np.interp(d, RHO_D, RHO_V)


def dir_to_pixel(dc, H, W):
    """camera-frame unit directions (N,3) -> (row, col) on the ERP grid, convention right0"""
    x, y, z = dc[:, 0], dc[:, 1], dc[:, 2]
    theta = np.arctan2(-x, -z); phi = np.arcsin(np.clip(y, -1, 1))
    u = 0.5 - theta / (2 * np.pi); v = 0.5 - phi / np.pi
    col = np.clip(np.floor(u * W).astype(int), 0, W - 1) % W; row = np.clip(np.floor(v * H).astype(int), 0, H - 1)
    return row, col, phi


def one(job):
    sc, sq = job; a = CFG; voxel, stride, md = a["voxel"], a["stride"], a["max_depth"]
    z = np.load(Path(a["pred_dir"]) / sc / f"{sq}.npz"); P = z["pred"].astype(np.float32); steps = z["steps"].tolist()
    S = Sequence(sc, sq); T = len(steps); H, W = P.shape[1:]; dirs = ray_dirs(H, W, a["convention"]); ds = dirs[::stride, ::stride]
    origins, Rs, clouds, gts, dpred_full, dgt_full, free_sets = [], [], [], [], [], [], []
    for k, i in enumerate(steps):
        pose = S.pose(i); R = quat_to_R(pose["rotation"]); o = np.asarray(pose["position"], float); origins.append(o); Rs.append(R)
        dp = to_radial(P[k], dirs, "face"); dpred_full.append(dp)
        d = dp[::stride, ::stride]; v = np.isfinite(d) & (d > 0) & (d < md); clouds.append((ds[v] * d[v][:, None]) @ R.T + o)
        gfull = to_radial(resize_nearest(S.gt_depth(i, "face"), (H, W)), dirs, "face"); dgt_full.append(np.where(np.isfinite(gfull) & (gfull > 0), gfull, np.inf))
        g = gfull[::stride, ::stride]; gv = np.isfinite(g) & (g > 0) & (g < md); gts.append((ds[gv] * g[gv][:, None]) @ R.T + o)
        dd = ds[v] @ R.T; dk = d[v] - 0.2; keys = []
        for j in range(int(np.ceil(max(dk.max(), 0) / voxel)) if dk.size else 0):
            t = (j + 0.5) * voxel; m = t < dk
            if not m.any():
                break
            keys.append(pack(o + dd[m] * t, voxel))
        free_sets.append(np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64))
    origins = np.stack(origins); u_ = origins[-1] - origins[0]; u_ /= max(np.linalg.norm(u_), 1e-9); proj = (origins - origins[0]) @ u_
    ref_fixed = voxelize(np.concatenate(gts), voxel / 2)[1]; nrm, rtree = normals(ref_fixed)
    r_ref = np.linalg.norm(ref_fixed[:, None, :] - origins[None, :, :], axis=2); jn = r_ref.argmin(1); r_ref = r_ref[np.arange(len(ref_fixed)), jn]
    vdir = ref_fixed - origins[jn]; vdir /= np.maximum(np.linalg.norm(vdir, axis=1, keepdims=True), 1e-9)
    inc_ref = np.degrees(np.arccos(np.clip(np.abs((vdir * nrm).sum(1)), 0, 1)))
    rbin = np.digitize(r_ref, [2, 4]); ibin = np.digitize(inc_ref, [30, 60]); ref_bin = rbin * 3 + ibin
    nref_bin = np.bincount(ref_bin, minlength=9)
    A = voxel ** 2; dOmega0 = 2 * np.pi ** 2 / ((H // stride) * (W // stride))   # solid angle of one ray at the working stride
    rows = []
    for N in NS:
        idx = subset(T, N, 0)
        ref_sub = voxelize(np.concatenate([gts[j] for j in idx]), voxel / 2)[1]; refs = {"fixed": ref_fixed, "subset": ref_sub}
        allP = np.concatenate([clouds[j] for j in idx]); view_of = np.concatenate([np.full(len(clouds[j]), j) for j in idx])
        uk, pos, cnt, inv = voxelize(allP, voxel); nv = len(uk)
        scores = {"count": cnt.astype(np.float64)}
        # ---- E108 A / A': expected rays from every view of the subset
        E_pred = np.zeros(nv); E_gt = np.zeros(nv)
        for j in idx:
            dv = pos - origins[j]; r = np.linalg.norm(dv, axis=1); dc = (dv / np.maximum(r[:, None], 1e-9)) @ Rs[j]   # world -> camera: R^T d  == d @ R
            row, col, phi = dir_to_pixel(dc, H, W)
            contrib = A / np.maximum(r ** 2 * dOmega0 * np.maximum(np.cos(phi), 1e-3), 1e-9)
            vis_p = r <= dpred_full[j][row, col] + 0.2; vis_g = r <= dgt_full[j][row, col] + 0.2
            E_pred += np.where(vis_p & (r < md), contrib, 0.0); E_gt += np.where(vis_g & (r < md), contrib, 0.0)
        for em in EMIN:
            scores[f"rate_geo_pred{em}"] = cnt / np.maximum(E_pred, em); scores[f"rate_geo_gt{em}"] = cnt / np.maximum(E_gt, em)
        # ---- E108 B: smoothed point density on a 0.25 m grid
        cell = 0.25; lo = allP.min(0) - 2.0; gi = np.floor((allP - lo) / cell).astype(int); shape = gi.max(0) + 9
        dens = np.zeros(shape); np.add.at(dens, (gi[:, 0], gi[:, 1], gi[:, 2]), 1.0)
        gv_ = np.floor((pos - lo) / cell).astype(int)
        for sg in SIGMAS:
            sm = gaussian_filter(dens, sg / cell, mode="constant") * (voxel / cell) ** 3
            scores[f"rate_smooth{sg}"] = cnt / np.maximum(sm[gv_[:, 0], gv_[:, 1], gv_[:, 2]], 1e-3)
        # ---- E109: (voxel, view) pairs in trajectory order
        pv = np.unique(inv.astype(np.int64) * 4096 + view_of); vox_of, vw = pv // 4096, pv % 4096
        first = np.r_[True, vox_of[1:] != vox_of[:-1]]
        d_prev = np.where(first, np.inf, np.r_[np.inf, np.abs(proj[vw][1:] - proj[vw][:-1])])
        w_a = np.where(first, 1.0, 1.0 - rho(d_prev))
        tie = 1e-3 * cnt                                     # ties (e.g. every weight 1 at N = 1) broken by the raw count
        scores["cont_a"] = np.bincount(vox_of, weights=w_a, minlength=nv) + tie
        # b) neighbour confirmation within the subset ordering
        pos_in_sub = {int(j): k for k, j in enumerate(idx)}; kk = np.array([pos_in_sub[int(j)] for j in vw])
        key_v = uk[vox_of]; offs = [np.array(o_) for o_ in ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
        pair_set = key_v * 4096 + kk
        for wdt in (1, 2):
            ok = np.zeros(len(pv), bool)
            for step in range(1, wdt + 1):
                for sgn in (-1, 1):
                    kk2 = kk + sgn * step
                    for o_ in offs:
                        nk = key_v + (o_[0] * 8192 + o_[1]) * 8192 + o_[2]
                        ok |= np.isin(nk * 4096 + kk2, pair_set)
            scores[f"cont_b{wdt}"] = np.bincount(vox_of, weights=ok.astype(float), minlength=nv) + tie
        # c) recursive log-odds with consecutive-view discount
        dstep = np.r_[np.inf, np.abs(np.diff(proj[idx]))]; w_seq = np.where(np.isinf(dstep), 1.0, 1.0 - rho(dstep))
        w_vote = np.bincount(vox_of, weights=w_seq[kk], minlength=nv)
        w_free = np.zeros(nv)
        for k_, j in enumerate(idx):
            fs = free_sets[j]
            if len(fs):
                h = np.minimum(np.searchsorted(fs, uk), len(fs) - 1); m_ = fs[h] == uk; w_free[m_] += w_seq[k_]
        for a_, b_ in CONT_C:
            scores[f"cont_c{a_}_{b_}"] = a_ * w_vote - b_ * w_free + tie
        # diagnostic: the same accumulations with rho(d) = exp(-d / lambda), lambda chosen on val
        for lam in LAMBDAS:
            w_a_e = np.where(first, 1.0, 1.0 - np.exp(-d_prev / lam))
            scores[f"cont_a_exp{lam}"] = np.bincount(vox_of, weights=w_a_e, minlength=nv) + tie
            w_seq_e = np.where(np.isinf(dstep), 1.0, 1.0 - np.exp(-dstep / lam))
            wv_e = np.bincount(vox_of, weights=w_seq_e[kk], minlength=nv); wf_e = np.zeros(nv)
            for k_, j in enumerate(idx):
                fs = free_sets[j]
                if len(fs):
                    h = np.minimum(np.searchsorted(fs, uk), len(fs) - 1); m_ = fs[h] == uk; wf_e[m_] += w_seq_e[k_]
            scores[f"cont_c_exp{lam}"] = wv_e - wf_e + tie
        base = dict(scene=sc, seq=sq, n_steps=T, N_req=(N if N > 0 else -1), N=len(idx), n_vox=nv)

        def add(name, keep, extra=None):
            for rn, ref in refs.items():
                m = metrics(pos[keep], ref, voxel=voxel)
                if rn == "fixed":
                    dk_, _ = cKDTree(pos[keep]).query(ref_fixed, k=1, workers=1)
                    hit = np.bincount(ref_bin, weights=dk_ < 0.2, minlength=9)
                    m.update({f"hit_b{b}": float(hit[b]) for b in range(9)}, **{f"nref_b{b}": int(nref_bin[b]) for b in range(9)})
                rows.append({**base, "ranking": name, "ref": rn, "kept": int(len(keep)), **(extra or {}), **m})

        for name, sc_ in scores.items():
            order = np.argsort(-sc_, kind="stable")
            for fr in FRACS:
                if fr == 1.0 and name != "count":
                    continue
                keep = order[: max(1, int(fr * nv))]; add(name, keep, {"frac": fr})
        # thresholded E108 rates, with the count ranking at the same kept count
        order_c = np.argsort(-cnt, kind="stable")
        for name in [k for k in scores if k.startswith("rate_geo_pred") or k.startswith("rate_smooth")]:
            for t in THRESH:
                keep = np.flatnonzero(scores[name] >= t)
                if len(keep) == 0:
                    continue
                add(name, keep, {"frac": -1.0, "thresh": t}); add("count_match_" + name, order_c[: len(keep)], {"frac": -1.0, "thresh": t})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="r2"); ap.add_argument("--split", default="test"); ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--voxel", type=float, default=0.1); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--convention", default="right0"); ap.add_argument("--workers", type=int, default=20); ap.add_argument("--seq-limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "E108_E109_continuity")
    a = ap.parse_args(); a.pred_dir = a.pred_dir or str(REPO / "outputs" / "pred" / a.mode); CFG.update({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()})
    scenes = TEST_SCENES if a.split == "test" else VAL_SCENES
    jobs = [(sc, sq) for sc in scenes for sq in (sequences(sc)[: a.seq_limit] if a.seq_limit else sequences(sc))]
    out = a.out / f"{a.mode}{'' if a.split == 'test' else '_val'}"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); rows = []
    with Pool(a.workers, initializer=lambda c: CFG.update(c), initargs=(dict(CFG),)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs)):
            rows += r; print(f"[{i+1}/{len(jobs)}] {len(rows)} rows {(time.time()-t0)/60:.1f} min", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(out / "per_sequence.csv", index=False)
    (out / "git_commit.txt").write_text(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout)
    print(f"{len(df)} rows in {(time.time()-t0)/60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
