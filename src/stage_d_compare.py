#!/usr/bin/env python3
"""Before/after table for Stage D: method x view count x run, with provenance.

Every cell is test F1@0.2 at the kept fraction chosen on the matching
validation run (when one exists; otherwise the test-side maximum, marked *).
Each run's directory and the commit recorded in its git_commit.txt are printed
so a reader can trace every number.

    python src/stage_d_compare.py --runs r2 r2_v3_whole r2_v3 --val r2_v3_val --mode r2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "E30_stage_d"
METHODS = ("A_point", "B_argmax", "C_on_B", "C_grid", "D_conf_sum", "E_conf_filter50", "B_argmax_matchE50",
           "E_conf_filter75", "B_argmax_matchE75", "oracle")


def table(run: str, val: str | None, metric: str):
    te = pd.read_csv(RES / run / "per_sequence.csv")
    va = pd.read_csv(RES / val / "per_sequence.csv") if val and (RES / val / "per_sequence.csv").exists() else None
    out, starred = {}, False
    for m in METHODS:
        for n in sorted(te.N_req.unique(), key=lambda n: (n < 0, n)):
            gt = te[(te.method == m) & (te.N_req == n)].groupby("frac")[metric].mean()
            if len(gt) == 0:
                continue
            if va is not None:
                gv = va[(va.method == m) & (va.N_req == n)].groupby("frac")[metric].mean()
                if len(gv):
                    f = float(gv.idxmax()); out[(m, int(n))] = float(gt.get(f, float("nan"))); continue
            out[(m, int(n))] = float(gt.max()); starred = True
    return out, starred


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--val", nargs="*", default=[], help="validation run per test run (same order; '-' for none)")
    ap.add_argument("--metric", default="f1@0.2")
    a = ap.parse_args()
    vals = a.val + ["-"] * (len(a.runs) - len(a.val))
    tabs = {}
    print("| run | directory | commit | temperature | band | frac chosen on |")
    print("|---|---|---|---|---|---|")
    for r, v in zip(a.runs, vals):
        d = RES / r
        commit = (d / "git_commit.txt").read_text().strip()[:7] if (d / "git_commit.txt").exists() else "?"
        cfg = {}
        if (d / "config.yaml").exists():
            import json; cfg = json.loads((d / "config.yaml").read_text())
        T = cfg.get("temperature_used", "1 (not applied)"); band = cfg.get("band", "whole (pre-fix)")
        tabs[r], starred = table(r, None if v == "-" else v, a.metric)
        print(f"| {r} | results/E30_stage_d/{r} | {commit} | {T if isinstance(T, str) else f'{T:.3f}'} | {band} | {'test max *' if starred else v} |")
    ns = sorted({n for t in tabs.values() for _, n in t}, key=lambda n: (n < 0, n))
    print(f"\n{a.metric}, test; columns = view count; one sub-column per run in the order given\n")
    print("| method | N | " + " | ".join(a.runs) + " |")
    print("|---|---|" + "---|" * len(a.runs))
    for m in METHODS:
        for n in ns:
            cells = [tabs[r].get((m, n)) for r in a.runs]
            if all(c is None for c in cells):
                continue
            print(f"| {m} | {'all' if n < 0 else n} | " + " | ".join("[미확인]" if c is None else f"{c:.3f}" for c in cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
