#!/usr/bin/env python3
"""Predict + fuse + score every sequence of the held-out scenes, then aggregate.

    python src/run_all.py --mode r2 --scenes apartment_2 frl_apartment_5 office_4 --gpu 3
writes outputs/summary_<mode>.md / .json (mean over sequences, per scene and overall).
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

PY = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--scenes", nargs="+", default=list(TEST_SCENES))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--hop", type=int, default=160)
    ap.add_argument("--dropout-samples", type=int, default=8)
    ap.add_argument("--step-stride", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    rows = []
    for sc in a.scenes:
        for sq in sequences(sc):
            pred = REPO / "outputs" / "pred" / a.mode / sc / f"{sq}.npz"
            fus = REPO / "outputs" / "fusion" / a.mode / sc / f"{sq}{a.tag}.json"
            if not (a.skip_existing and pred.exists()):
                subprocess.run([PY, str(HERE / "predict.py"), "--scene", sc, "--seq", sq, "--mode", a.mode, "--gpu", a.gpu,
                                "--hop", str(a.hop), "--dropout-samples", str(a.dropout_samples)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not (a.skip_existing and fus.exists()):
                subprocess.run([PY, str(HERE / "eval_fusion.py"), "--scene", sc, "--seq", sq, "--mode", a.mode,
                                "--step-stride", str(a.step_stride), "--tag", a.tag], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            r = json.loads(fus.read_text())
            rows.append((sc, sq, r))
            print(f"[{sc}/{sq}] steps {r['n_steps']} ERP MAE {r['erp_mae_m']:.2f} single acc {r['single_step']['acc']:.2f} "
                  f"fused-all acc {r['fused']['uniform']['voxel_avg']['acc']:.2f} top25 {r['fused']['uniform']['kept']['0.25']['acc']:.2f}", flush=True)

    def agg(sel):
        def m(f):
            return float(np.mean([f(r) for _, _, r in sel]))
        out = {"n_seq": len(sel), "erp_mae": m(lambda r: r["erp_mae_m"]),
               "single": {k: m(lambda r, k=k: r["single_step"][k]) for k in ("acc", "acc_frac", "comp")}}
        for rule in sel[0][2]["fused"]:
            out[rule] = {"all": {k: m(lambda r, k=k: r["fused"][rule]["voxel_avg"][k]) for k in ("acc", "acc_frac", "comp")}}
            for frac in ("0.75", "0.5", "0.25"):
                out[rule][frac] = {k: m(lambda r, k=k: r["fused"][rule]["kept"][frac][k]) for k in ("acc", "acc_frac", "comp")}
            out[rule]["tsdf"] = {k: m(lambda r, k=k: r["fused"][rule]["tsdf"][k]) for k in ("acc", "acc_frac", "comp")}
        return out

    S = {sc: agg([r for r in rows if r[0] == sc]) for sc in a.scenes}
    S["all"] = agg(rows)
    L = [f"# Fusion over the held-out sequences, mode {a.mode}, hop {a.hop}, every {a.step_stride} step(s)\n",
         "Mean over sequences. acc = mean nearest distance predicted->reference (m) / fraction within 0.2 m; comp = reference->predicted (m).\n",
         "| scene | seqs | ERP MAE | single acc | fused all acc | uniform top50 / top25 | range-prior top50 / top25 | model-std top50 / top25 | comp all / top25 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for sc, o in S.items():
        f = lambda d: f"{d['acc']:.2f}/{100*d['acc_frac']:.0f}%"
        L.append(f"| {sc} | {o['n_seq']} | {o['erp_mae']:.2f} | {f(o['single'])} | {f(o['uniform']['all'])} | "
                 f"{f(o['uniform']['0.5'])} / {f(o['uniform']['0.25'])} | {f(o['range_prior']['0.5'])} / {f(o['range_prior']['0.25'])} | "
                 f"{f(o['model_std']['0.5']) if 'model_std' in o else '–'} / {f(o['model_std']['0.25']) if 'model_std' in o else '–'} | "
                 f"{o['uniform']['all']['comp']:.2f} / {o['uniform']['0.25']['comp']:.2f} |")
    (REPO / "outputs" / f"summary_{a.mode}{a.tag}.md").write_text("\n".join(L) + "\n")
    (REPO / "outputs" / f"summary_{a.mode}{a.tag}.json").write_text(json.dumps(S, indent=1))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
