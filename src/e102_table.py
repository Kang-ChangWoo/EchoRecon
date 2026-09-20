#!/usr/bin/env python3
"""E102 table: baseline-aware support rankings vs the point-count control, per CRITERIA_E102.md.
Δ (and, separately, the kept fraction) is chosen on val by F1@0.2 at N = all (fixed reference);
test is read at that choice.

    python src/e102_table.py --mode r2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from e100_table import boot_ci  # noqa: E402

BAND = 0.03


RESULTS = ["E102_baseline_support"]


def load(mode, tag, split):
    f = REPO / "results" / RESULTS[0] / f"{mode}{tag}{'' if split == 'test' else '_val'}" / "per_sequence.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f); df["sid"] = df.scene + "/" + df.seq
    return df


def piv(df, ranking, frac, ref="fixed", metric="f1@0.2", seed=0):
    d = df[(df.ranking == ranking) & (df.frac == frac) & (df.ref == ref) & (df.seed == seed)]
    return d.pivot_table(index="sid", columns="N_req", values=metric)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--tag", default="")
    ap.add_argument("--results-dir", default="E102_baseline_support")
    a = ap.parse_args()
    RESULTS[0] = a.results_dir
    df = load(a.mode, a.tag, "test"); dv = load(a.mode, a.tag, "val")
    if df is None:
        print("no test results"); return 1
    rankings = [r for r in df.ranking.unique() if r != "support"]
    print(f"=== E102 {a.mode}{a.tag}: {df.sid.nunique()} test sequences, val {'yes' if dv is not None else 'none'}")
    # val choice: best Δ among cells* at frac 0.25, N=all; and best frac for support and for the chosen cells
    chosen = None
    if dv is not None:
        best = {}
        for r in [x for x in rankings if x.startswith("cells")]:
            best[r] = piv(dv, r, 0.25)[-1].mean()
        chosen = max(best, key=best.get)
        fr_best = {r: max((0.5, 0.25, 0.1), key=lambda f: piv(dv, r, f)[-1].mean()) for r in ("support", chosen)}
        print("val: F1(all, frac .25) per Δ: " + ", ".join(f"{k}: {v:.3f}" for k, v in best.items()) + f" -> chosen {chosen}; best fraction on val: {fr_best}")
    print("\n## test, fixed reference, frac 0.25, seed 0: F1@0.2 by N (P/R at N=all)")
    hdr = [1, 2, 4, 8, 16, -1]
    for r in ["support"] + sorted(rankings):
        c = piv(df, r, 0.25); p = piv(df, r, 0.25, metric="precision@0.2"); q = piv(df, r, 0.25, metric="recall@0.2")
        mark = " <- chosen" if r == chosen else ""
        print(f"{r:10s} " + " ".join(f"{c[n].mean():.3f}" for n in hdr) + f"   P/R(all) {p[-1].mean():.3f}/{q[-1].mean():.3f}{mark}")
    print("\n## gain over support (paired, fixed ref, frac 0.25): N=4 / 8 / 16 / all, CI at all, verdict")
    s = piv(df, "support", 0.25)
    for r in sorted(rankings):
        c = piv(df, r, 0.25); g = {n: (c[n] - s[n]).dropna() for n in (4, 8, 16, -1)}
        ci = boot_ci(g[-1]); gm = g[-1].mean()
        v = "meaningful" if gm >= BAND and ci[0] > 0 else ("worse" if gm <= -BAND and ci[1] < 0 else ("no effect" if abs(gm) < BAND else "inside CI"))
        per = df[(df.ranking == r) & (df.frac == 0.25) & (df.ref == "fixed") & (df.seed == 0) & (df.N_req == -1)].set_index("sid")["f1@0.2"] - \
              df[(df.ranking == "support") & (df.frac == 0.25) & (df.ref == "fixed") & (df.seed == 0) & (df.N_req == -1)].set_index("sid")["f1@0.2"]
        per = per.rename("v").reset_index(); per["scene"] = per.sid.str.split("/").str[0]
        print(f"{r:10s} " + " ".join(f"{g[n].mean():+.3f}" for n in (4, 8, 16, -1)) + f"  CI[{ci[0]:+.3f},{ci[1]:+.3f}]  {v:11s} per scene " +
              ", ".join(f"{sc}:{x.v.mean():+.3f}±{x.v.std():.3f}" for sc, x in per.groupby("scene")))
    for fr in (0.5, 0.1):
        print(f"\n## frac {fr}: F1(all) support {piv(df, 'support', fr)[-1].mean():.3f}; " + ", ".join(f"{r}: {piv(df, r, fr)[-1].mean():.3f}" for r in sorted(rankings)))
    print("\n## subset reference, frac 0.25, F1 at N=4/16/all: " + "; ".join(f"{r}: " + "/".join(f"{piv(df, r, 0.25, ref='subset')[n].mean():.3f}" for n in (4, 16, -1)) for r in ["support"] + sorted(rankings)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
