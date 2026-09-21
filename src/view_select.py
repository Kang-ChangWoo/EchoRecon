#!/usr/bin/env python3
"""E102b (issue B2): distance-based view selection, the direct experiment. From all views of a
sequence adopt greedily only views >= Δ from every adopted view; controls at the same count:
a random subset and the consecutive block centred in the sequence. A_point pipeline, fixed
reference (subset reference recorded), frac 1.0 and 0.25. Criteria: results/E100_cause/CRITERIA_B_C.md.

    python src/view_select.py --mode r2
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

DELTAS = (0.3, 0.5, 1.0)
FRACS = (1.0, 0.25)
CFG = {}


def greedy_spaced(origins, delta):
    keep = [0]
    for j in range(1, len(origins)):
        if np.linalg.norm(origins[keep] - origins[j], axis=1).min() >= delta:
            keep.append(j)
    return np.array(keep)


def one(job):
    sc, sq = job; a = CFG; voxel, stride, md = a["voxel"], a["stride"], a["max_depth"]
    z = np.load(Path(a["pred_dir"]) / sc / f"{sq}.npz"); P = z["pred"].astype(np.float32); steps = z["steps"].tolist()
    S = Sequence(sc, sq); T = len(steps); H, W = P.shape[1:]; dirs = ray_dirs(H, W, a["convention"]); ds = dirs[::stride, ::stride]
    origins, clouds, gts = [], [], []
    for k, i in enumerate(steps):
        pose = S.pose(i); R = quat_to_R(pose["rotation"]); o = np.asarray(pose["position"], float); origins.append(o)
        d = to_radial(P[k], dirs, "face")[::stride, ::stride]; v = np.isfinite(d) & (d > 0) & (d < md); clouds.append((ds[v] * d[v][:, None]) @ R.T + o)
        g = to_radial(resize_nearest(S.gt_depth(i, "face"), (H, W)), dirs, "face")[::stride, ::stride]; gv = np.isfinite(g) & (g > 0) & (g < md); gts.append((ds[gv] * g[gv][:, None]) @ R.T + o)
    origins = np.stack(origins); ref_fixed = voxelize(np.concatenate(gts), voxel / 2)[1]
    rng = np.random.default_rng(0)
    rows = []

    def score(name, idx, extra):
        allP = np.concatenate([clouds[j] for j in idx]); _, pos, cnt, _ = voxelize(allP, voxel)
        ref_sub = voxelize(np.concatenate([gts[j] for j in idx]), voxel / 2)[1]
        span = float(np.linalg.norm(origins[idx].max(0) - origins[idx].min(0)))
        order = np.argsort(-cnt, kind="stable")
        for fr in FRACS:
            keep = order[: max(1, int(fr * len(pos)))]
            for rn, ref in (("fixed", ref_fixed), ("subset", ref_sub)):
                rows.append(dict(scene=sc, seq=sq, n_steps=T, selection=name, n=len(idx), span_m=span, frac=fr, ref=rn, **extra, **metrics(pos[keep], ref, voxel=voxel)))

    score("all", np.arange(T), {"delta": 0.0})
    for D in DELTAS:
        idx = greedy_spaced(origins, D); n = len(idx)
        score(f"spaced{D}", idx, {"delta": D})
        score(f"random_n{D}", np.sort(rng.choice(T, n, replace=False)), {"delta": D})
        s0 = (T - n) // 2; score(f"consec_n{D}", np.arange(s0, s0 + n), {"delta": D})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="r2"); ap.add_argument("--split", default="test"); ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--voxel", type=float, default=0.1); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--convention", default="right0"); ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "E102b_view_select")
    a = ap.parse_args(); a.pred_dir = a.pred_dir or str(REPO / "outputs" / "pred" / a.mode); CFG.update({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()})
    scenes = TEST_SCENES if a.split == "test" else VAL_SCENES
    jobs = [(sc, sq) for sc in scenes for sq in sequences(sc)]
    out = a.out / f"{a.mode}{'' if a.split == 'test' else '_val'}"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); rows = []
    with Pool(a.workers, initializer=lambda c: CFG.update(c), initargs=(dict(CFG),)) as pool:
        for r in pool.imap_unordered(one, jobs):
            rows += r
    import pandas as pd
    from e100_table import boot_ci
    df = pd.DataFrame(rows); df["sid"] = df.scene + "/" + df.seq; df.to_csv(out / "per_sequence.csv", index=False)
    (out / "git_commit.txt").write_text(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout)
    lines = [f"# E102b {a.mode} {a.split}: {df.sid.nunique()} sequences ({(time.time()-t0)/60:.1f} min)"]
    for ref in ("fixed", "subset"):
        for fr in FRACS:
            d = df[(df.ref == ref) & (df.frac == fr)]
            f_all = d[d.selection == "all"].set_index("sid")["f1@0.2"]
            lines.append(f"\n## ref {ref}, frac {fr}: F1(all views) = {f_all.mean():.3f} (n = {d[d.selection=='all'].n.mean():.1f})")
            for D in DELTAS:
                sp = d[d.selection == f"spaced{D}"].set_index("sid"); ra = d[d.selection == f"random_n{D}"].set_index("sid")["f1@0.2"]; co = d[d.selection == f"consec_n{D}"].set_index("sid")["f1@0.2"]
                g_r = (sp["f1@0.2"] - ra.loc[sp.index]); g_c = (sp["f1@0.2"] - co.loc[sp.index]); g_a = (sp["f1@0.2"] - f_all.loc[sp.index])
                ci = boot_ci(g_r)
                v = "restores" if g_r.mean() >= 0.03 and ci[0] > 0 else ("no effect" if abs(g_r.mean()) < 0.03 else "inside CI")
                per = ", ".join(f"{s_}:{gg.mean():+.3f}" for s_, gg in g_r.groupby(g_r.index.str.split('/').str[0]))
                lines.append(f"Δ={D}: n={sp.n.mean():.1f} span {sp.span_m.mean():.2f} m | spaced {sp['f1@0.2'].mean():.3f} (P {sp['precision@0.2'].mean():.3f} R {sp['recall@0.2'].mean():.3f}) "
                             f"random-same-n {ra.mean():.3f} consec-same-n {co.mean():.3f} | spaced−random {g_r.mean():+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] {v}; spaced−consec {g_c.mean():+.3f}; spaced−all {g_a.mean():+.3f} | per scene (vs random) {per}")
    txt = "\n".join(lines); print(txt); (out / "summary.txt").write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
