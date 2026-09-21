#!/usr/bin/env python3
"""E100: cause decomposition of the falling view-count curve (no training, CPU only).

Criteria are pre-registered in results/E100_cause/CRITERIA.md. One pass per
sequence: unproject every step once, then evaluate every (N, seed, variant,
reference) on that material.

  reference  subset    GT of the selected N steps, fused (what Stage A/D used)
             fixed     GT of every step of the sequence, fused (N-independent)     -> cause (e)
  variant    base      the A_point pipeline as in Stage A
             compact   N consecutive steps centred in the sequence, instead of the
                       evenly spread subset                                          -> cause (b)
             smooth<w> per-view median filter (w px) on the ERP depth before fusion  -> cause (c) control
             contra<k> voxels dropped when other views saw free space through them:
                       drop if c >= k*s (k=0: c >= 1)                                -> cause (d)
plus measurements: reference growth (e), viewing-angle spread vs precision (b),
signed-error correlation within a view on the same GT plane and across views
at the same surface voxel (c), wrong rate of contradicted vs uncontradicted
voxels (d).

    python src/cause_decomp.py --mode r2 --split test --out results/E100_cause
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
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from data import TEST_SCENES, VAL_SCENES, Sequence, sequences  # noqa: E402
from erp import quat_to_R, ray_dirs, to_radial  # noqa: E402
from eval_fusion import resize_nearest  # noqa: E402
from fuse import metrics  # noqa: E402
from stage_a import subset  # noqa: E402

NS = (1, 2, 4, 8, 16, 0)
FRACS = (1.0, 0.25)
SMOOTH = (5, 9, 15)
CONTRA = (0, 1, 2)
CFG = {}
OFF, MUL = 4096, 8192          # packed voxel key: (ix+OFF)*MUL^2 + (iy+OFF)*MUL + (iz+OFF)


def pack(P: np.ndarray, voxel: float) -> np.ndarray:
    k = np.floor(P / voxel).astype(np.int64) + OFF
    return (k[:, 0] * MUL + k[:, 1]) * MUL + k[:, 2]


def voxelize(P: np.ndarray, voxel: float):
    """sorted unique packed keys, mean position per voxel, point count, inverse index"""
    key = pack(P, voxel)
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    pos = np.stack([np.bincount(inv, weights=P[:, d], minlength=len(uk)) for d in range(3)], 1) / cnt[:, None]
    return uk, pos, cnt, inv


def compact_subset(n_steps: int, N: int) -> np.ndarray:
    if N <= 0 or N >= n_steps:
        return np.arange(n_steps)
    s = (n_steps - N) // 2
    return np.arange(s, s + N)


def corr_stats(x, y):
    """sufficient statistics of a Pearson correlation, poolable by addition"""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    return dict(n=int(len(x)), sx=float(x.sum()), sy=float(y.sum()), sxx=float((x * x).sum()),
                syy=float((y * y).sum()), sxy=float((x * y).sum()))


def pearson(s):
    n = s["n"]
    if n < 3:
        return float("nan")
    cov = s["sxy"] / n - (s["sx"] / n) * (s["sy"] / n)
    vx = s["sxx"] / n - (s["sx"] / n) ** 2; vy = s["syy"] / n - (s["sy"] / n) ** 2
    return float(cov / np.sqrt(max(vx * vy, 1e-18)))


def add_stats(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def gt_normals(Pc: np.ndarray, valid: np.ndarray, d: np.ndarray):
    """normals of the GT camera-frame point image (H, W, 3) by finite differences; invalid on
    depth discontinuities (> 0.3 m between neighbours) and at invalid pixels."""
    H, W = valid.shape
    Pr = np.roll(Pc, -1, 1) - np.roll(Pc, 1, 1)      # along azimuth (wraps)
    Pu = np.roll(Pc, -1, 0) - np.roll(Pc, 1, 0)      # along elevation
    n = np.cross(Pr, Pu)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-9)
    ok = valid & np.roll(valid, -1, 1) & np.roll(valid, 1, 1) & np.roll(valid, -1, 0) & np.roll(valid, 1, 0)
    jump = np.maximum.reduce([np.abs(np.roll(d, -1, 1) - d), np.abs(np.roll(d, 1, 1) - d),
                              np.abs(np.roll(d, -1, 0) - d), np.abs(np.roll(d, 1, 0) - d)])
    ok &= jump < 0.3
    ok[0] = ok[-1] = False
    return n, ok


def one_sequence(job):
    sc, sq = job
    a = CFG
    voxel, stride, md = a["voxel"], a["stride"], a["max_depth"]
    pf = Path(a["pred_dir"]) / sc / f"{sq}.npz"
    if not pf.exists():
        return [], {}
    z = np.load(pf)
    P_face = z["pred"].astype(np.float32); steps = z["steps"].tolist()
    S = Sequence(sc, sq)
    T = len(steps)
    H, W = P_face.shape[1:]
    dirs = ray_dirs(H, W, a["convention"])
    dirs_s = dirs[::stride, ::stride]
    rng = np.random.default_rng(0)

    # ---- per-step material
    origins, Rs = [], []
    gt_rad_s, gt_valid_s, gt_world, gt_cam, gt_norm, gt_normok = [], [], [], [], [], []
    pred_rad = {}                     # variant -> list of radial depth (stride) per step
    pred_rad["base"] = []
    for w in SMOOTH:
        pred_rad[f"smooth{w}"] = []
    for k, i in enumerate(steps):
        pose = S.pose(i); R = quat_to_R(pose["rotation"]); o = np.asarray(pose["position"], float)
        origins.append(o); Rs.append(R)
        g = resize_nearest(S.gt_depth(i, "face"), (H, W))
        g_rad = to_radial(g, dirs, "face")
        gv = np.isfinite(g_rad) & (g_rad > 0) & (g_rad < md)
        Pc_full = dirs * np.where(gv, g_rad, 0)[..., None]
        n_full, nok_full = gt_normals(Pc_full, gv, np.where(gv, g_rad, 0))
        gs, gvs = g_rad[::stride, ::stride], gv[::stride, ::stride]
        gt_rad_s.append(gs); gt_valid_s.append(gvs)
        gt_cam.append(Pc_full[::stride, ::stride]); gt_norm.append(n_full[::stride, ::stride]); gt_normok.append(nok_full[::stride, ::stride])
        gt_world.append((dirs_s[gvs] * gs[gvs][:, None]) @ R.T + o)
        pred_rad["base"].append(to_radial(P_face[k], dirs, "face")[::stride, ::stride])
        for w in SMOOTH:
            pad = w // 2   # wrap in azimuth, clamp in elevation
            padded = np.concatenate([P_face[k][:, -pad:], P_face[k], P_face[k][:, :pad]], 1)
            sm = median_filter(padded, size=(w, w), mode="nearest")[:, pad:-pad]
            pred_rad[f"smooth{w}"].append(to_radial(sm, dirs, "face")[::stride, ::stride])
    origins = np.stack(origins)

    def cloud(var, k):
        d = pred_rad[var][k]
        v = np.isfinite(d) & (d > 0) & (d < md)
        return (dirs_s[v] * d[v][:, None]) @ Rs[k].T + origins[k]

    clouds = {var: [cloud(var, k) for k in range(T)] for var in pred_rad}
    ref_fixed_pts = np.concatenate(gt_world)
    _, ref_fixed, _, _ = voxelize(ref_fixed_pts, voxel / 2)

    # ---- (d) per-view free-space voxel sets from the base prediction, margin along the ray
    margin, fs_step = a["margin"], voxel
    free_sets, own_sets = [], []
    for k in range(T):
        d = pred_rad["base"][k]
        v = np.isfinite(d) & (d > 0) & (d < md)
        dd = dirs_s[v] @ Rs[k].T; dk = d[v] - margin
        nmax = int(np.ceil(dk.max() / fs_step)) if dk.size else 0
        keys = []
        for j in range(nmax):
            t = (j + 0.5) * fs_step
            m = t < dk
            if not m.any():
                break
            keys.append(pack(origins[k] + dd[m] * t, voxel))
        free_sets.append(np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64))
        own_sets.append(np.unique(pack(clouds["base"][k], voxel)))

    # ---- (c) measurements, once per sequence (independent of N)
    meas = {"scene": sc, "seq": sq, "n_steps": T}
    sep_classes = {"1-2deg": (0.7, 1.5), "5-10deg": (3.5, 7.0), "20-30deg": (14.0, 21.0)}
    cstats = {f"{sep}|{pl}|{cen}": {} for sep in sep_classes for pl in ("same", "diff") for cen in ("raw", "centred")}
    Hs, Ws = gt_rad_s[0].shape
    rows_ok = np.zeros(Hs, bool); rows_ok[Hs // 4: 3 * Hs // 4] = True      # |elevation| < 45 deg
    for k in range(T):
        e = pred_rad["base"][k] - gt_rad_s[k]
        val = gt_valid_s[k] & gt_normok[k] & np.isfinite(e)
        ec = e - e[val].mean() if val.any() else e
        anchors = np.flatnonzero(val & rows_ok[:, None])
        if len(anchors) < 100:
            continue
        pick = rng.choice(anchors, min(a["pairs_per_view"], len(anchors)), replace=False)
        r0, c0 = np.divmod(pick, Ws)
        for sep, (lo, hi) in sep_classes.items():
            ang = rng.uniform(0, 2 * np.pi, len(pick)); rad = rng.uniform(lo, hi, len(pick))
            r1 = np.clip(np.round(r0 + rad * np.sin(ang)).astype(int), 0, Hs - 1)
            c1 = np.mod(np.round(c0 + rad * np.cos(ang)).astype(int), Ws)
            ok = val[r1, c1] & ((r1 != r0) | (c1 != c0))
            if not ok.any():
                continue
            A = (r0[ok], c0[ok]); B = (r1[ok], c1[ok])
            nA, nB = gt_norm[k][A], gt_norm[k][B]
            PA, PB = gt_cam[k][A], gt_cam[k][B]
            ndot = np.abs((nA * nB).sum(1)); pdist = np.abs(((PB - PA) * nA).sum(1))
            same = (ndot > 0.95) & (pdist < 0.05)
            diff = (ndot < 0.7) | (pdist > 0.3)
            for pl, m in (("same", same), ("diff", diff)):
                if m.sum() < 3:
                    continue
                cstats[f"{sep}|{pl}|raw"] = add_stats(cstats[f"{sep}|{pl}|raw"], corr_stats(e[A][m], e[B][m]))
                cstats[f"{sep}|{pl}|centred"] = add_stats(cstats[f"{sep}|{pl}|centred"], corr_stats(ec[A][m], ec[B][m]))
    meas["within_view"] = cstats
    # cross-view: two views' signed errors at the same GT surface voxel (0.2 m), by baseline distance
    xv_vox = 0.2
    keys_all, err_all, errc_all, view_all = [], [], [], []
    for k in range(T):
        e = pred_rad["base"][k] - gt_rad_s[k]
        val = gt_valid_s[k] & np.isfinite(e)
        gw = (dirs_s[val] * gt_rad_s[k][val][:, None]) @ Rs[k].T + origins[k]
        keys_all.append(pack(gw, xv_vox)); ev = e[val]
        err_all.append(ev); errc_all.append(ev - ev.mean()); view_all.append(np.full(len(ev), k))
    keys_all = np.concatenate(keys_all); err_all = np.concatenate(err_all); errc_all = np.concatenate(errc_all); view_all = np.concatenate(view_all)
    # one (voxel, view) entry: mean error of that view's rays in the voxel
    kv = keys_all * 4096 + view_all
    ukv, inv = np.unique(kv, return_inverse=True); inv = inv.ravel()
    cnt = np.bincount(inv); me = np.bincount(inv, weights=err_all) / cnt; mec = np.bincount(inv, weights=errc_all) / cnt
    uk = ukv // 4096; uv = ukv % 4096
    # consecutive entries with the same voxel key: pair the first with each later one
    same_prev = np.flatnonzero(uk[1:] == uk[:-1]) + 1
    # walk back to the first entry of each run
    first = np.arange(len(uk)); 
    for _ in range(64):
        prev_same = (first > 0) & (uk[np.maximum(first - 1, 0)] == uk)
        if not prev_same.any():
            break
        first = np.where(prev_same, first - 1, first)
    pi = first[same_prev]; pj = same_prev
    bdist = np.linalg.norm(origins[uv[pi]] - origins[uv[pj]], axis=1)
    xstats = {}
    for name, m in (("all", np.ones(len(pi), bool)), ("<0.5m", bdist < 0.5), ("0.5-1.5m", (bdist >= 0.5) & (bdist < 1.5)), (">=1.5m", bdist >= 1.5)):
        if m.sum() >= 3:
            xstats[f"{name}|raw"] = corr_stats(me[pi][m], me[pj][m]); xstats[f"{name}|centred"] = corr_stats(mec[pi][m], mec[pj][m])
    meas["cross_view"] = xstats

    # ---- per (N, seed): references, variants, metrics
    rows = []
    d_meas = {"n_vox": 0, "n_contra": 0, "wrong_contra": 0, "wrong_uncontra": 0}
    b_meas = {}
    for N in NS:
        for seed in range(a["seeds"]):
            if N <= 0 and seed > 0:
                continue
            idx = subset(T, N, seed)
            ref_sub_pts = np.concatenate([gt_world[j] for j in idx])
            _, ref_sub, _, _ = voxelize(ref_sub_pts, voxel / 2)
            refs = {"subset": ref_sub, "fixed": ref_fixed}
            span = float(np.linalg.norm(origins[idx].max(0) - origins[idx].min(0)))
            base = dict(scene=sc, seq=sq, n_steps=T, N_req=(N if N > 0 else -1), N=len(idx), seed=seed, span_m=span,
                        n_ref_subset=len(ref_sub), n_ref_fixed=len(ref_fixed))

            def score(variant, pts, cnt, ref_name, extra=None):
                ref = refs[ref_name]
                order = np.argsort(-cnt)
                for fr in FRACS:
                    keep = order[: max(1, int(fr * len(pts)))]
                    m = metrics(pts[keep], ref, voxel=voxel)
                    rows.append({**base, "variant": variant, "ref": ref_name, "frac": fr, **(extra or {}), **m})

            # base and the smoothing variants
            for var in pred_rad:
                allP = np.concatenate([clouds[var][j] for j in idx])
                if len(allP) == 0:
                    continue
                uk_v, pos, cnt, inv = voxelize(allP, voxel)
                for rn in refs:
                    score(var, pos, cnt, rn)
                if var != "base":
                    continue
                # (d) contradiction counts on the base voxels
                fs_cat = np.concatenate([free_sets[j] for j in idx])
                fu, fc = np.unique(fs_cat, return_counts=True)
                c_free = np.zeros(len(uk_v), np.int64)
                hit = np.searchsorted(fu, uk_v); hit = np.minimum(hit, len(fu) - 1)
                m_ = fu[hit] == uk_v if len(fu) else np.zeros(len(uk_v), bool)
                c_free[m_] = fc[hit[m_]]
                # views that both saw through the voxel and put a point in it do not count as contradicting
                both = np.zeros(len(uk_v), np.int64)
                for j in idx:
                    b_ = np.intersect1d(free_sets[j], own_sets[j], assume_unique=True)
                    if len(b_):
                        h = np.searchsorted(b_, uk_v); h = np.minimum(h, len(b_) - 1)
                        both += (b_[h] == uk_v)
                c = np.maximum(c_free - both, 0)
                # support s = number of distinct views with a point in the voxel (the ranking
                # keeps the Stage A point count, s is for the contradiction rule and the spread bins)
                view_of_pt = np.concatenate([np.full(len(clouds[var][j]), j) for j in idx])
                pv = np.unique(inv.astype(np.int64) * 4096 + view_of_pt)
                s = np.bincount(pv // 4096, minlength=len(uk_v))
                # (b) viewing-angle spread per voxel: dispersion of the unit vectors from the
                # contributing views' origins to the voxel centre
                dv = pos[inv] - origins[view_of_pt]; dv /= np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-9)
                mean_dir = np.stack([np.bincount(inv, weights=dv[:, d_], minlength=len(uk_v)) for d_ in range(3)], 1) / cnt[:, None]
                spread_deg = np.degrees(np.arccos(np.clip(np.linalg.norm(mean_dir, axis=1), -1, 1)))
                for k_ in CONTRA:
                    keep = c < np.maximum(k_ * s, 1) if k_ > 0 else (c < 1)
                    if keep.sum() == 0:
                        continue
                    for rn in refs:
                        score(f"contra{k_}", pos[keep], cnt[keep], rn, {"kept_frac_vox": float(keep.mean())})
                    # budget-matched control: the same number of voxels kept by support rank
                    # alone (Stage A's ranking), so the contradiction rule is read against
                    # "keep the well-supported voxels" and not against "keep fewer voxels"
                    top = np.argsort(-cnt, kind="stable")[: int(keep.sum())]
                    for rn in refs:
                        score(f"support_match{k_}", pos[top], cnt[top], rn, {"kept_frac_vox": float(keep.mean())})
                if N <= 0 and seed == 0:
                    # existence measurements at N = all, fixed reference
                    err, _ = cKDTree(ref_fixed).query(pos, k=1, workers=1)
                    wrong = err > 0.2; contra = c >= 1
                    d_meas.update(n_vox=int(len(pos)), n_contra=int(contra.sum()), wrong_contra=int((wrong & contra).sum()),
                                  wrong_uncontra=int((wrong & ~contra).sum()), n_uncontra=int((~contra).sum()),
                                  mean_c=float(c.mean()), mean_s=float(s.mean()),
                                  # multi-view voxels only (s >= 2), so single-view voxels do not drive the rate
                                  n_mv=int((s >= 2).sum()), n_mv_contra=int((contra & (s >= 2)).sum()),
                                  wrong_mv_contra=int((wrong & contra & (s >= 2)).sum()), wrong_mv_uncontra=int((wrong & ~contra & (s >= 2)).sum()))
                    # (b) precision by spread tertile within matched support bins
                    sbins = [(2, 2), (3, 4), (5, 8), (9, 10 ** 6)]
                    for lo_, hi_ in sbins:
                        m_ = (s >= lo_) & (s <= hi_)
                        if m_.sum() < 30:
                            continue
                        q = np.quantile(spread_deg[m_], [1 / 3, 2 / 3])
                        low = m_ & (spread_deg <= q[0]); high = m_ & (spread_deg >= q[1])
                        b_meas[f"s{lo_}-{hi_}"] = dict(n_low=int(low.sum()), right_low=int((~wrong & low).sum()),
                                                        n_high=int(high.sum()), right_high=int((~wrong & high).sum()),
                                                        spread_q=[float(q[0]), float(q[1])])
            # (b) compact subset, base variant only
            if 0 < N < T:
                idc = compact_subset(T, N)
                allP = np.concatenate([clouds["base"][j] for j in idc])
                _, pos, cnt, _ = voxelize(allP, voxel)
                ref_c = voxelize(np.concatenate([gt_world[j] for j in idc]), voxel / 2)[1]
                spanc = float(np.linalg.norm(origins[idc].max(0) - origins[idc].min(0)))
                for rn, ref in (("subset", ref_c), ("fixed", ref_fixed)):
                    order = np.argsort(-cnt)
                    for fr in FRACS:
                        keep = order[: max(1, int(fr * len(pos)))]
                        m = metrics(pos[keep], ref, voxel=voxel)
                        rows.append({**base, "variant": "compact", "ref": rn, "frac": fr, "span_m": spanc,
                                     "n_ref_subset": len(ref_c), **m})
    meas["contradiction"] = d_meas
    meas["spread_precision"] = b_meas
    return rows, meas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--pred-dir", default=None, help="default outputs/pred/<mode>")
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--margin", type=float, default=0.2, help="free space stops this far before the predicted surface")
    ap.add_argument("--convention", default="right0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pairs-per-view", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seq-limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "E100_cause")
    a = ap.parse_args()
    a.pred_dir = a.pred_dir or str(REPO / "outputs" / "pred" / a.mode)
    CFG.update({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()})
    out = a.out / f"{a.mode}{a.tag}{'' if a.split == 'test' else '_val'}"
    out.mkdir(parents=True, exist_ok=True)
    scenes = a.scenes or list(TEST_SCENES if a.split == "test" else VAL_SCENES)
    jobs = [(sc, sq) for sc in scenes for sq in (sequences(sc)[: a.seq_limit] if a.seq_limit else sequences(sc))]
    t0 = time.time()
    rows, meas = [], []
    with Pool(a.workers, initializer=lambda c: CFG.update(c), initargs=(dict(CFG),)) as pool:
        for i, (r, m) in enumerate(pool.imap_unordered(one_sequence, jobs)):
            rows += r
            if m:
                meas.append(m)
            print(f"[{i+1}/{len(jobs)}] {len(rows)} rows, {(time.time()-t0)/60:.1f} min", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_sequence.csv", index=False)
    (out / "measurements.json").write_text(json.dumps(meas, indent=1))
    (out / "config.yaml").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    (out / "git_commit.txt").write_text(
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout)
    print(f"{len(df)} rows, {len(meas)} sequences in {(time.time()-t0)/60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
