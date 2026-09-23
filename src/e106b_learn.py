#!/usr/bin/env python3
"""E106b: the per-voxel learner fitted on the TRAIN scenes' features, hyper-parameters chosen on
val (small grid: hidden {32,64,128} x epochs {20,40} x features {all, no confidence}), read once on
test. The chosen learner is then applied unchanged to other heads' test voxels (transfer).

    CUDA_VISIBLE_DEVICES=1 python src/e106b_learn.py --mode r2 --eval r2_test r2_warm_test r2_warm_headonly_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from e100_table import boot_ci  # noqa: E402
from e106_learn import auroc, prep, ref_for  # noqa: E402
from fuse import metrics  # noqa: E402

FD = REPO / "results" / "E106_learned" / "features"
GRID = [(h, e, f) for h in (32, 64, 128) for e in (20, 40) for f in ("all", "noconf")]


def load(stem):
    z = np.load(FD / f"{stem}.npz", allow_pickle=True); return list(z["rows"]), list(z["feats"])


def train_mlp(X, y, hidden, epochs, device, seed=0, bs=8192):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32, device=device); yt = torch.tensor(y, dtype=torch.float32, device=device)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)).to(device)
    pw = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-6), device=device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5); lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]; opt.zero_grad(); loss = lossf(net(Xt[b]).squeeze(1), yt[b]); loss.backward(); opt.step()
    net.eval(); return net


def evaluate(net, rows, feats, sel, mu, sd, frac, device, refs):
    """per (seq, N): F1 etc. of support and of the learner at the top fraction, fixed reference"""
    ci = feats.index("count"); out = []
    with torch.no_grad():
        for r in rows:
            X = (prep(r["X"], feats) - mu) / sd
            sc_l = net(torch.tensor(X[:, sel], dtype=torch.float32, device=device)).squeeze(1).cpu().numpy()
            key = (r["scene"], r["seq"])
            if key not in refs:
                refs[key] = ref_for(*key)
            ref = refs[key]; k = max(1, int(frac * len(r["y"])))
            for name, sc in (("support", r["X"][:, ci].astype(np.float64)), ("learner", sc_l)):
                order = np.argsort(-sc, kind="stable")[:k]
                out.append(dict(scene=r["scene"], seq=r["seq"], N_req=r["N_req"], ranking=name, auroc=auroc(sc, r["y"]), **metrics(r["pos"][order], ref, voxel=0.1)))
    import pandas as pd
    return pd.DataFrame(out)


def gain_table(df, N=-1):
    d = df[df.N_req == N]; s = d[d.ranking == "support"].set_index(["scene", "seq"]); l = d[d.ranking == "learner"].set_index(["scene", "seq"]).loc[s.index]
    g = l["f1@0.2"] - s["f1@0.2"]; ci = boot_ci(g)
    return dict(n=len(g), f1_support=float(s["f1@0.2"].mean()), f1_learner=float(l["f1@0.2"].mean()), P=float(l["precision@0.2"].mean()), R=float(l["recall@0.2"].mean()),
                auroc_support=float(s.auroc.mean()), auroc_learner=float(l.auroc.mean()), gain=float(g.mean()), ci_lo=ci[0], ci_hi=ci[1],
                per_scene={sc: (float(gg.mean()), float(gg.std())) for sc, gg in g.groupby(level=0)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="r2"); ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--eval", nargs="+", default=None, help="test feature stems to read at the val-chosen config (default <mode>_test)")
    ap.add_argument("--fit", default="train", choices=("train", "val"))
    ap.add_argument("--feats-fixed", default=None, choices=(None, "all", "noconf"), help="restrict the grid to one feature set (E111: ranker without / with the head's aggregates)")
    ap.add_argument("--tag", default="", help="suffix on the result directory")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows_fit, feats = load(f"{a.mode}_{a.fit}"); rows_val, _ = load(f"{a.mode}_val")
    X = np.concatenate([r["X"] for r in rows_fit]); y = np.concatenate([r["y"] for r in rows_fit]).astype(np.float64)
    Xp = prep(X, feats); mu, sd = Xp.mean(0), Xp.std(0) + 1e-9; Xn = (Xp - mu) / sd
    yv = np.concatenate([r["y"] for r in rows_val])
    od = REPO / "results" / "E106b_trainfit" / f"{a.mode}{a.tag}"; od.mkdir(parents=True, exist_ok=True)
    lines = [f"# E106b {a.mode}: fit on {a.fit} scenes ({len(rows_fit)} (seq,N) rows, {len(y)} voxels, positive rate {y.mean():.3f}); val {len(rows_val)} rows, {len(yv)} voxels, positive rate {yv.mean():.3f}"]
    refs = {}; grid_res = []
    grid = [g for g in GRID if a.feats_fixed is None or g[2] == a.feats_fixed]
    for h, e, f in grid:
        sel = [i for i, ff in enumerate(feats) if f == "all" or not ff.startswith("conf")]
        net = train_mlp(Xn[:, sel], y, h, e, device)
        dv = evaluate(net, rows_val, feats, sel, mu, sd, a.frac, device, refs); gt = gain_table(dv)
        grid_res.append(dict(hidden=h, epochs=e, feats=f, val_gain=gt["gain"], val_f1=gt["f1_learner"], val_auroc=gt["auroc_learner"]))
        lines.append(f"grid hidden {h:3d} epochs {e:2d} feats {f:6s}: val gain {gt['gain']:+.3f} (F1 {gt['f1_learner']:.3f} vs support {gt['f1_support']:.3f}, AUROC {gt['auroc_learner']:.3f})")
        print(lines[-1], flush=True)
    best = max(grid_res, key=lambda r: r["val_gain"]); lines.append(f"val choice: {best}")
    sel = [i for i, ff in enumerate(feats) if best["feats"] == "all" or not ff.startswith("conf")]
    net = train_mlp(Xn[:, sel], y, best["hidden"], best["epochs"], device)
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd, "sel": sel, "feats": feats, "config": best}, od / "learner.pt")
    for stem in (a.eval or [f"{a.mode}_test"]):
        rows_t, _ = load(stem)
        dt = evaluate(net, rows_t, feats, sel, mu, sd, a.frac, device, refs); dt.to_csv(od / f"per_sequence_{stem}.csv", index=False)
        for N in (-1, 4):
            g = gain_table(dt, N)
            lines.append(f"TEST {stem:26s} N={'all' if N < 0 else N:>3}: learner F1 {g['f1_learner']:.3f} (P {g['P']:.3f} R {g['R']:.3f}) vs support {g['f1_support']:.3f} | gain {g['gain']:+.3f} CI[{g['ci_lo']:+.3f},{g['ci_hi']:+.3f}] "
                         f"| AUROC {g['auroc_learner']:.3f} vs {g['auroc_support']:.3f} | per scene " + ", ".join(f"{k}:{v[0]:+.3f}±{v[1]:.3f}" for k, v in g["per_scene"].items()))
            print(lines[-1], flush=True)
    (od / "grid.json").write_text(json.dumps(grid_res, indent=1)); (od / "summary.txt").write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
