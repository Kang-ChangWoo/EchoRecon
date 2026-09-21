#!/usr/bin/env python3
"""E106: train a small MLP on the val scenes' per-voxel features (voxel_features.py) to predict
"within 0.2 m of the fixed reference", use it as a voxel ranking on the test scenes, and compare
with support (count) and a count-only logistic control at the top 25 %. Criteria:
results/E100_cause/CRITERIA_E104_E106.md.

    python src/e106_learn.py --mode r2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from cause_decomp import voxelize  # noqa: E402
from data import Sequence  # noqa: E402
from e100_table import boot_ci  # noqa: E402
from erp import quat_to_R, ray_dirs, to_radial  # noqa: E402
from eval_fusion import resize_nearest  # noqa: E402
from fuse import metrics  # noqa: E402

LOGF = {"count", "views", "cells0.3", "cells0.6", "cells1.0", "contra", "N", "range_mean", "range_min"}


def prep(X, feats):
    X = X.astype(np.float64).copy()
    for i, f in enumerate(feats):
        if f in LOGF:
            X[:, i] = np.log1p(np.maximum(X[:, i], 0))
    return X


def auroc(score, y):
    order = np.argsort(score); r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]; _, first, cnt = np.unique(s_sorted, return_index=True, return_counts=True)
    for f, c in zip(first, cnt):
        if c > 1:
            r[order[f:f + c]] = f + (c + 1) / 2
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))


def train_mlp(X, y, hidden=64, epochs=40, seed=0, device="cpu"):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32, device=device); yt = torch.tensor(y, dtype=torch.float32, device=device)
    d = X.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)).to(device) \
        if hidden > 0 else torch.nn.Linear(d, 1).to(device)
    pw = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-6), device=device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5); lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(Xt); bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]; opt.zero_grad(); loss = lossf(net(Xt[b]).squeeze(1), yt[b]); loss.backward(); opt.step()
    net.eval()
    return net


def ref_for(scene, seq, voxel=0.1, stride=2, md=10.0):
    S = Sequence(scene, seq); dirs = None; pts = []
    for i in S.steps:
        g = S.gt_depth(i, "face"); H, W = 256, 512
        if dirs is None:
            dirs = ray_dirs(H, W); ds = dirs[::stride, ::stride]
        gr = to_radial(resize_nearest(g, (H, W)), dirs, "face")[::stride, ::stride]; gv = np.isfinite(gr) & (gr > 0) & (gr < md)
        pose = S.pose(i); R = quat_to_R(pose["rotation"]); o = np.asarray(pose["position"], float)
        pts.append((ds[gv] * gr[gv][:, None]) @ R.T + o)
    return voxelize(np.concatenate(pts), voxel / 2)[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="r2"); ap.add_argument("--frac", type=float, default=0.25); ap.add_argument("--device", default="cpu")
    ap.add_argument("--group", default="all", choices=("all", "single", "inter"),
                    help="feature group for the ablation: single = per-ray geometry only (elevation, range, count, N); "
                         "inter = between-view statistics only (views, cells, span, spread, contra, frac_views, count, N)")
    a = ap.parse_args()
    fd = REPO / "results" / "E106_learned" / "features"
    va = np.load(fd / f"{a.mode}_val.npz", allow_pickle=True); te = np.load(fd / f"{a.mode}_test.npz", allow_pickle=True)
    feats = list(va["feats"]); rows_v = list(va["rows"]); rows_t = list(te["rows"])
    Xv = np.concatenate([r["X"] for r in rows_v]); yv = np.concatenate([r["y"] for r in rows_v]).astype(np.float64)
    Xv = prep(Xv, feats); mu, sd = Xv.mean(0), Xv.std(0) + 1e-9; Xv = (Xv - mu) / sd
    ci_ = feats.index("count")
    print(f"=== E106 {a.mode}: val voxels {len(yv)} (pos {yv.mean():.3f}), test (seq,N) rows {len(rows_t)}; features {feats}")
    GROUPS = {"single": ["abs_elev_mean", "range_mean", "range_min", "count", "N"],
              "inter": ["views", "cells0.3", "cells0.6", "cells1.0", "span", "spread_deg", "contra", "frac_views", "count", "N"]}
    if a.group != "all":
        sel = [feats.index(f) for f in GROUPS[a.group]]
    else:
        sel = list(range(len(feats)))
    mlp = train_mlp(Xv[:, sel], yv, 64, device=a.device)
    logit_cnt = train_mlp(Xv[:, [ci_]], yv, 0, device=a.device)
    geo_idx = [i for i in sel if not feats[i].startswith("conf")]
    mlp_geo = train_mlp(Xv[:, geo_idx], yv, 64, device=a.device)   # no confidence
    out_rows = []; refs = {}
    per = {}
    with torch.no_grad():
        for r in rows_t:
            X = (prep(r["X"], feats) - mu) / sd; Xt = torch.tensor(X, dtype=torch.float32, device=a.device)
            scores = {"support": r["X"][:, ci_].astype(np.float64),
                      "logit_count": logit_cnt(Xt[:, [ci_]]).squeeze(1).cpu().numpy(),
                      "mlp_all": mlp(Xt[:, sel]).squeeze(1).cpu().numpy(),
                      "mlp_noconf": mlp_geo(Xt[:, geo_idx]).squeeze(1).cpu().numpy()}
            key = (r["scene"], r["seq"])
            if key not in refs:
                refs[key] = ref_for(*key)
            ref = refs[key]; k = max(1, int(a.frac * len(r["y"])))
            for name, sc in scores.items():
                order = np.argsort(-sc, kind="stable")[:k]
                m = metrics(r["pos"][order], ref, voxel=0.1)
                out_rows.append(dict(scene=r["scene"], seq=r["seq"], N_req=r["N_req"], ranking=name, auroc=auroc(sc, r["y"]), prec_top=float(r["y"][order].mean()), **m))
    import pandas as pd
    df = pd.DataFrame(out_rows); od = REPO / "results" / "E106_learned" / (a.mode if a.group == "all" else f"{a.mode}_{a.group}"); od.mkdir(parents=True, exist_ok=True); df.to_csv(od / "per_sequence.csv", index=False)
    lines = [f"# E106 {a.mode} group={a.group} features={[feats[i] for i in sel]}"]
    for N in (4, -1):
        d = df[df.N_req == N]; s = d[d.ranking == "support"].set_index(["scene", "seq"])
        lines.append(f"\n## N={'all' if N < 0 else N}, top {a.frac:.0%}, fixed reference, {s.shape[0]} sequences")
        for name in ("support", "logit_count", "mlp_noconf", "mlp_all"):
            x = d[d.ranking == name].set_index(["scene", "seq"]).loc[s.index]
            g = (x["f1@0.2"] - s["f1@0.2"]); ci = boot_ci(g)
            v = "meaningful" if g.mean() >= 0.03 and ci[0] > 0 else ("no effect" if abs(g.mean()) < 0.03 else "inside CI")
            per_scene = ", ".join(f"{sc}:{gg.mean():+.3f}±{gg.std():.3f}" for sc, gg in g.groupby(level=0))
            lines.append(f"{name:12s} F1 {x['f1@0.2'].mean():.3f} P {x['precision@0.2'].mean():.3f} R {x['recall@0.2'].mean():.3f} AUROC {x['auroc'].mean():.3f} | gain {g.mean():+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] {v} | {per_scene}")
    # permutation importance of the full MLP on pooled test voxels at N=all (descriptive)
    Xt_all = np.concatenate([r["X"] for r in rows_t if r["N_req"] == -1]); yt_all = np.concatenate([r["y"] for r in rows_t if r["N_req"] == -1])
    Xn = (prep(Xt_all, feats) - mu) / sd
    with torch.no_grad():
        base = auroc(mlp(torch.tensor(Xn[:, sel], dtype=torch.float32)).squeeze(1).numpy(), yt_all)
        imp = {}
        rng = np.random.default_rng(0)
        for i in sel:
            Xp = Xn.copy(); Xp[:, i] = rng.permutation(Xp[:, i])
            imp[feats[i]] = base - auroc(mlp(torch.tensor(Xp[:, sel], dtype=torch.float32)).squeeze(1).numpy(), yt_all)
    lines.append(f"\nMLP AUROC on pooled test voxels (N=all): {base:.3f}; permutation importance (AUROC drop): " + ", ".join(f"{k}:{v:.3f}" for k, v in sorted(imp.items(), key=lambda kv: -kv[1])))
    txt = "\n".join(lines); print(txt); (od / "summary.txt").write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
