#!/usr/bin/env python3
"""Choose the kept fraction on the validation sequences, then read the test number.

Stage A-D print "best F1 over the kept-fraction sweep", which picks the fraction
per method on the test sequences themselves. That is symmetric across methods
but still a test-side choice. This script selects, per method and view count,
the fraction with the highest mean F1@0.2 on the validation run and reports the
test F1 at that fraction, next to the test-side maximum so the size of the
optimism is visible.

    python src/select_frac.py --val results/E30_stage_d/r2_Tfit_val --test results/E30_stage_d/r2_Tfit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METHODS = ("A_point", "B_argmax", "C_on_B", "C_grid", "D_conf_sum", "E_conf_filter50", "B_argmax_matchE50",
           "E_conf_filter75", "B_argmax_matchE75", "oracle")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--metric", default="f1@0.2")
    a = ap.parse_args()
    va = pd.read_csv(a.val / "per_sequence.csv"); te = pd.read_csv(a.test / "per_sequence.csv")
    ns = sorted(te.N_req.unique(), key=lambda n: (n < 0, n))
    out = {}
    print(f"{a.metric}: test at the val-selected fraction (test-side max in brackets)")
    print(f"{'method':16s}" + "".join(f"{('all' if n < 0 else n):>16}" for n in ns))
    for m in METHODS:
        row = {}
        for n in ns:
            gv = va[(va.method == m) & (va.N_req == n)].groupby("frac")[a.metric].mean()
            gt = te[(te.method == m) & (te.N_req == n)].groupby("frac")[a.metric].mean()
            if len(gv) == 0 or len(gt) == 0:
                continue
            f = float(gv.idxmax())
            row[int(n)] = dict(frac=f, val=float(gv.max()), test=float(gt.get(f, float("nan"))), test_max=float(gt.max()))
        out[m] = row
        print(f"{m:16s}" + "".join(f"{row[int(n)]['test']:8.3f} [{row[int(n)]['test_max']:.3f}]" if int(n) in row else " " * 16 for n in ns))
    (a.test / "val_selected.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {a.test / 'val_selected.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
