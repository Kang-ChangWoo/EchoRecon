#!/usr/bin/env python3
"""E107 (issue C): what the per-view median-confidence filter removes. On the posterior head's
B cloud at N = all: every voxel is kept or lost by the filter; per voxel correct (< 0.2 m from the
fixed reference), range (mean distance to its views), incidence (angle between the mean viewing
direction and the reference normal at the nearest reference point). Criteria: CRITERIA_B_C.md.

    python src/conf_loss_decomp.py --mode r2
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
from data import TEST_SCENES, Sequence, sequences  # noqa: E402
from erp import quat_to_R, ray_dirs, to_radial  # noqa: E402
from eval_fusion import resize_nearest  # noqa: E402

RB = [(0, 2), (2, 4), (4, 99)]; IB = [(0, 30), (30, 60), (60, 91)]
CFG = {}


def normals(ref, k=16):
    tree = cKDTree(ref); _, nn = tree.query(ref, k=k, workers=1)
    P = ref[nn] - ref[nn].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", P, P)
    w, v = np.linalg.eigh(C)
    return v[:, :, 0], tree      # smallest-eigenvalue eigenvector


def one(job):
    sc, sq = job; a = CFG; voxel, stride, md, q = a["voxel"], a["stride"], a["max_depth"], a["quantile"]
    z = np.load(Path(a["pred_dir"]) / sc / f"{sq}.npz"); P = z["pred"].astype(np.float32); C = z["conf"].astype(np.float32); steps = z["steps"].tolist()
    S = Sequence(sc, sq); T = len(steps); H, W = P.shape[1:]; dirs = ray_dirs(H, W, a["convention"]); ds = dirs[::stride, ::stride]
    origins, pts, confs, gts, dview = [], [], [], [], []
    for k, i in enumerate(steps):
        pose = S.pose(i); R = quat_to_R(pose["rotation"]); o = np.asarray(pose["position"], float); origins.append(o)
        d = to_radial(P[k], dirs, "face")[::stride, ::stride]; v = np.isfinite(d) & (d > 0) & (d < md)
        pts.append((ds[v] * d[v][:, None]) @ R.T + o); confs.append(C[k][::stride, ::stride][v]); dview.append(ds[v] @ R.T)
        g = to_radial(resize_nearest(S.gt_depth(i, "face"), (H, W)), dirs, "face")[::stride, ::stride]; gv = np.isfinite(g) & (g > 0) & (g < md); gts.append((ds[gv] * g[gv][:, None]) @ R.T + o)
    origins = np.stack(origins); ref = voxelize(np.concatenate(gts), voxel / 2)[1]
    nrm, tree = normals(ref)
    allP = np.concatenate(pts); allC = np.concatenate(confs); allD = np.concatenate(dview)
    view_of = np.concatenate([np.full(len(p), j) for j, p in enumerate(pts)])
    keep_pt = np.concatenate([c >= np.quantile(c, q) for c in confs])
    uk, pos, cnt, inv = voxelize(allP, voxel); nv = len(uk)
    kept_vox = np.bincount(inv, weights=keep_pt, minlength=nv) > 0
    err, nn_ = tree.query(pos, k=1, workers=1); correct = err < 0.2
    rng_ = np.bincount(inv, weights=np.linalg.norm(allP - origins[view_of], axis=1), minlength=nv) / cnt
    mdir = np.stack([np.bincount(inv, weights=allD[:, d_], minlength=nv) for d_ in range(3)], 1); mdir /= np.maximum(np.linalg.norm(mdir, axis=1, keepdims=True), 1e-9)
    inc = np.degrees(np.arccos(np.clip(np.abs((mdir * nrm[nn_]).sum(1)), 0, 1)))
    rb = np.digitize(rng_, [2, 4]); ib = np.digitize(inc, [30, 60])
    # counts per (range bin, incidence bin): all, correct, lost, lost&correct
    key = rb * 3 + ib
    tab = {}
    for name, m in (("all", np.ones(nv, bool)), ("correct", correct), ("lost", ~kept_vox), ("lost_correct", ~kept_vox & correct), ("kept_correct", kept_vox & correct)):
        tab[name] = np.bincount(key[m], minlength=9).reshape(3, 3).tolist()
    # recall per range bin of the reference (range = distance to the nearest view origin), full vs kept cloud
    d_full, _ = cKDTree(pos).query(ref, k=1, workers=1); d_kept, _ = cKDTree(pos[kept_vox]).query(ref, k=1, workers=1) if kept_vox.any() else (np.full(len(ref), np.inf), None)
    r_ref = np.linalg.norm(ref[:, None, :] - origins[None, :, :], axis=2).min(1); rbin = np.digitize(r_ref, [2, 4])
    rec = {"n_ref": np.bincount(rbin, minlength=3).tolist(), "hit_full": np.bincount(rbin, weights=d_full < 0.2, minlength=3).tolist(), "hit_kept": np.bincount(rbin, weights=d_kept < 0.2, minlength=3).tolist()}
    return dict(scene=sc, seq=sq, n_vox=int(nv), n_kept=int(kept_vox.sum()), table=tab, recall=rec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="r2"); ap.add_argument("--pred-dir", default=None); ap.add_argument("--quantile", type=float, default=0.5)
    ap.add_argument("--voxel", type=float, default=0.1); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--convention", default="right0"); ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args(); a.pred_dir = a.pred_dir or str(REPO / "outputs" / "pred" / f"{a.mode}_post"); CFG.update(vars(a))
    jobs = [(sc, sq) for sc in TEST_SCENES for sq in sequences(sc)]
    t0 = time.time(); res = []
    with Pool(a.workers, initializer=lambda c: CFG.update(c), initargs=(dict(CFG),)) as pool:
        for r in pool.imap_unordered(one, jobs):
            res.append(r)
    out = REPO / "results" / "E107_conf_paradox"; out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.mode}_q{int(a.quantile*100)}.json").write_text(json.dumps(res, indent=1))
    A = lambda name: np.sum([np.array(r["table"][name]) for r in res], 0)
    allv, cor, lost, lc, kc = A("all"), A("correct"), A("lost"), A("lost_correct"), A("kept_correct")
    lines = [f"# E107 {a.mode}, filter = per-view confidence below the {a.quantile:.0%} quantile dropped, N = all, {len(res)} sequences",
             f"voxels {allv.sum():.0f}, lost {lost.sum()/allv.sum():.1%}; correct {cor.sum()/allv.sum():.1%}; lost among correct {lc.sum()/cor.sum():.1%}",
             "rows = range {<2, 2-4, >=4 m}, cols = incidence {<30, 30-60, >=60 deg}"]
    with np.errstate(invalid="ignore", divide="ignore"):
        lines.append("share of ALL voxels per bin:\n" + np.array2string(allv / allv.sum(), precision=3))
        lines.append("share of LOST-CORRECT voxels per bin (where the lost right answers were):\n" + np.array2string(lc / max(lc.sum(), 1), precision=3))
        lines.append("fraction of correct voxels LOST per bin (the 'sole source' rate):\n" + np.array2string(lc / np.maximum(cor, 1), precision=3))
        lines.append("precision (correct/all) per bin, full cloud:\n" + np.array2string(cor / np.maximum(allv, 1), precision=3))
    far_or_obl = lc[2, :].sum() + lc[:2, 2].sum(); near_frontal_rate = lc[0, 0] / max(cor[0, 0], 1)
    far_rate = lc[2, :].sum() / max(cor[2, :].sum(), 1); obl_rate = lc[:, 2].sum() / max(cor[:, 2].sum(), 1)
    lines.append(f"(i) share of lost-correct voxels that are far (>=4 m) or oblique (>=60 deg): {far_or_obl / max(lc.sum(), 1):.3f}")
    lines.append(f"(ii) lost fraction of correct voxels: near&frontal {near_frontal_rate:.3f}, far {far_rate:.3f} ({far_rate/max(near_frontal_rate,1e-9):.1f}x), oblique {obl_rate:.3f} ({obl_rate/max(near_frontal_rate,1e-9):.1f}x)")
    nref = np.sum([r["recall"]["n_ref"] for r in res], 0); hf = np.sum([r["recall"]["hit_full"] for r in res], 0); hk = np.sum([r["recall"]["hit_kept"] for r in res], 0)
    lines.append("(iii) recall@0.2 of the reference by range bin {<2, 2-4, >=4 m}: full cloud " + np.array2string(hf / np.maximum(nref, 1), precision=3) + ", after the filter " + np.array2string(hk / np.maximum(nref, 1), precision=3) + f" (reference share per bin {np.array2string(nref / nref.sum(), precision=2)})")
    per_scene = {}
    for sc in TEST_SCENES:
        rs = [r for r in res if r["scene"] == sc]
        lc_ = np.sum([np.array(r["table"]["lost_correct"]) for r in rs], 0); cor_ = np.sum([np.array(r["table"]["correct"]) for r in rs], 0)
        per_scene[sc] = f"(i) {(lc_[2,:].sum()+lc_[:2,2].sum())/max(lc_.sum(),1):.2f}, far rate {lc_[2,:].sum()/max(cor_[2,:].sum(),1):.2f} vs near&frontal {lc_[0,0]/max(cor_[0,0],1):.2f}"
    lines.append("per scene: " + "; ".join(f"{k}: {v}" for k, v in per_scene.items()))
    txt = "\n".join(lines); print(txt); (out / f"{a.mode}_q{int(a.quantile*100)}_summary.txt").write_text(txt); print(f"{(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
