#!/usr/bin/env python3
"""Predict + fuse + score every sequence of the held-out scenes, then aggregate.

    python src/run_all.py --mode r2 --gpu 3                     # all steps of each sequence
    python src/run_all.py --mode r2 --gpu 3 --n-views 4 --tag _N4   # four views per sequence

writes outputs/summary_<mode><tag>.md / .json: the mean over sequences of the
single-step accuracy, of each ranking's top-fraction curve, and of the recovery
rate against the oracle ranking.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from data import TEST_SCENES, sequences  # noqa: E402
from eval_fusion import FRACS  # noqa: E402

PY = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--scenes", nargs="+", default=list(TEST_SCENES))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--hop", type=int, default=160)
    ap.add_argument("--dropout-samples", type=int, default=8)
    ap.add_argument("--step-stride", type=int, default=1)
    ap.add_argument("--n-views", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    rows = []
    for sc in a.scenes:
        for sq in sequences(sc):
            pred = REPO / "outputs" / "pred" / a.mode / sc / f"{sq}.npz"
            fus = REPO / "outputs" / "fusion" / a.mode / sc / f"{sq}{a.tag}.json"
            if not pred.exists():
                subprocess.run([PY, str(HERE / "predict.py"), "--scene", sc, "--seq", sq, "--mode", a.mode,
                                "--gpu", a.gpu, "--hop", str(a.hop),
                                "--dropout-samples", str(a.dropout_samples)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not (a.skip_existing and fus.exists()):
                subprocess.run([PY, str(HERE / "eval_fusion.py"), "--scene", sc, "--seq", sq, "--mode", a.mode,
                                "--step-stride", str(a.step_stride), "--n-views", str(a.n_views),
                                "--tag", a.tag], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            r = json.loads(fus.read_text())
            rows.append(r)
            print(f"[{sc}/{sq}] N={r['n_steps']:3d} ERP MAE {r['erp_mae_m']:.2f} single {r['single_step']['acc']:.2f} "
                  f"fused-all {r['fused']['uniform']['all']['acc']:.2f} "
                  f"top25 support {r['fused']['uniform']['kept']['0.25']['acc']:.2f} "
                  f"oracle {r['fused']['oracle']['kept']['0.25']['acc']:.2f}", flush=True)

    def m(f):
        vals = [f(r) for r in rows]
        return float(np.nanmean(vals))

    rules = [k for k in rows[0]["fused"]]
    S = {"n_seq": len(rows), "mode": a.mode, "tag": a.tag,
         "n_views_mean": m(lambda r: r["n_steps"]), "erp_mae": m(lambda r: r["erp_mae_m"]),
         "single": {k: m(lambda r, k=k: r["single_step"][k]) for k in ("acc", "acc_frac", "comp")}}
    for rule in rules:
        S[rule] = {"all": {k: m(lambda r, k=k, u=rule: r["fused"][u]["all"][k]) for k in ("acc", "acc_frac", "comp")}}
        for f in FRACS:
            S[rule][str(f)] = {k: m(lambda r, k=k, u=rule, f=f: r["fused"][u]["kept"][str(f)][k])
                               for k in ("acc", "acc_frac", "comp")}
    S["recovery"] = {rule: {str(f): m(lambda r, u=rule, f=f: r["recovery"][u][str(f)]) for f in FRACS}
                     for rule in rows[0]["recovery"]}

    def cell(d):
        return f"{d['acc']:.2f}/{100*d['acc_frac']:.0f}%"

    L = [f"# Fusion over the held-out sequences: mode {a.mode}, hop {a.hop}, "
         f"every {a.step_stride} step(s), N = {S['n_views_mean']:.1f} views on average\n",
         f"Mean over {len(rows)} sequences. acc = mean nearest distance predicted -> reference (m) and the "
         f"fraction within tau; comp = reference -> predicted (m). Voxel positions are identical across "
         f"rankings; only the order differs.\n",
         "| ranking | all voxels | top 75% | top 50% | top 25% | top 10% | comp all / top 25% |",
         "|---|---|---|---|---|---|---|",
         f"| single step (no fusion) | {cell(S['single'])} | | | | | {S['single']['comp']:.2f} / – |"]
    for rule in rules:
        L.append(f"| {rule} | {cell(S[rule]['all'])} | " + " | ".join(cell(S[rule][str(f)]) for f in FRACS)
                 + f" | {S[rule]['all']['comp']:.2f} / {S[rule]['0.25']['comp']:.2f} |")
    L += ["", "Recovery rate (uniform - method) / (uniform - oracle): 0 = no better than counting points, "
              "1 = as good as knowing the answer.\n",
          "| ranking | top 75% | top 50% | top 25% | top 10% |", "|---|---|---|---|---|"]
    for rule, v in S["recovery"].items():
        L.append(f"| {rule} | " + " | ".join(f"{v[str(f)]:+.2f}" for f in FRACS) + " |")
    (REPO / "outputs" / f"summary_{a.mode}{a.tag}.md").write_text("\n".join(L) + "\n")
    (REPO / "outputs" / f"summary_{a.mode}{a.tag}.json").write_text(json.dumps(S, indent=1))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
