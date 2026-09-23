#!/usr/bin/env python3
"""Tables for E108/E109 (continuity_fusion.py) and E110 (topk_fusion.py): val-chosen settings, test
gains vs the baseline (count / B_argmax) with paired bootstrap CIs, N-curves, slopes, and recall of
the fixed reference by range and incidence bin. Criteria: CRITERIA_E108_E110.md.

    python src/e108_table.py --exp E108_E109_continuity --mode r2
    python src/e108_table.py --exp E110_topk --mode r2
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
RB = ["<2m", "2-4m", ">=4m"]; IB = ["<30", "30-60", ">=60"]


def load(exp, mode, split):
    f = REPO / "results" / exp / f"{mode}{'' if split == 'test' else '_val'}" / "per_sequence.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f); df["sid"] = df.scene + "/" + df.seq; return df


def piv(df, ranking, frac, ref="fixed", metric="f1@0.2", thresh=None):
    d = df[(df.ranking == ranking) & (df.ref == ref)]
    d = d[d.frac == frac] if thresh is None else d[(d.frac == -1.0) & (np.isclose(d.thresh, thresh))]
    return d.pivot_table(index="sid", columns="N_req", values=metric)


def recall_bins(df, ranking, frac, N=-1, thresh=None):
    d = df[(df.ranking == ranking) & (df.ref == "fixed") & (df.N_req == N)]
    d = d[d.frac == frac] if thresh is None else d[(d.frac == -1.0) & (np.isclose(d.thresh, thresh))]
    hit = np.array([d[f"hit_b{b}"].sum() for b in range(9)]); n = np.array([d[f"nref_b{b}"].sum() for b in range(9)])
    r = hit / np.maximum(n, 1); byr = hit.reshape(3, 3).sum(1) / np.maximum(n.reshape(3, 3).sum(1), 1); byi = hit.reshape(3, 3).sum(0) / np.maximum(n.reshape(3, 3).sum(0), 1)
    return byr, byi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E108_E109_continuity"); ap.add_argument("--mode", default="r2"); ap.add_argument("--frac", type=float, default=0.25)
    a = ap.parse_args()
    df = load(a.exp, a.mode, "test"); dv = load(a.exp, a.mode, "val")
    if df is None:
        print("no test results"); return 1
    base = "count" if "count" in set(df.ranking) else "B_argmax"
    rankings = [r for r in df.ranking.unique() if r != base and not r.startswith("count_match_")]
    lines = [f"=== {a.exp} {a.mode}: {df.sid.nunique()} test sequences, val {'yes' if dv is not None else 'none'}, baseline {base}, frac {a.frac}"]
    b = piv(df, base, a.frac); bR, bI = recall_bins(df, base, a.frac)
    lines.append(f"{base:22s} F1 N=1/2/4/8/16/all " + "/".join(f"{b[n].mean():.3f}" for n in (1, 2, 4, 8, 16, -1)) + f" | slope(all-4) {b[-1].mean()-b[4].mean():+.3f} | recall by range {np.array2string(bR, precision=3)} by incidence {np.array2string(bI, precision=3)}")
    # families: choose the best member on val (frac a.frac, N = all, fixed) per family prefix
    fam = {}
    for r in rankings:
        key = r.rstrip("0123456789._") if any(ch.isdigit() for ch in r) else r
        if r.startswith("cont_a_exp"): key = "cont_a_exp"
        elif r.startswith("cont_c_exp"): key = "cont_c_exp"
        elif r.startswith("cont_c"): key = "cont_c"
        elif r.startswith("cont_b"): key = "cont_b"
        key = {"rate_geo_pred": "rate_geo_pred", "rate_geo_gt": "rate_geo_gt", "rate_smooth": "rate_smooth", "topk_count": "topk_count", "topk_mass": "topk_mass"}.get(key, key)
        fam.setdefault(key, []).append(r)
    chosen = {}
    for key, members in fam.items():
        if dv is not None and len(members) > 1:
            bv = piv(dv, base, a.frac)
            best = max(members, key=lambda r: (piv(dv, r, a.frac)[-1] - bv[-1]).mean() if -1 in piv(dv, r, a.frac).columns else -9)
        else:
            best = members[0]
        chosen[key] = best
    lines.append("val choice per family: " + ", ".join(f"{k}: {v}" for k, v in chosen.items()))
    lines.append(f"\n## test, fixed ref, frac {a.frac}: gain vs {base} at N=all [CI], N=4, slope difference, curve, recall by range / incidence (N=all)")
    for key, r in chosen.items():
        c = piv(df, r, a.frac)
        if c.empty or -1 not in c.columns:
            continue
        g = (c[-1] - b[-1]).dropna(); ci = boot_ci(g); g4 = (c[4] - b[4]).dropna()
        slope_d = (c[-1] - c[4]).mean() - (b[-1] - b[4]).mean()
        v = "meaningful" if g.mean() >= BAND and ci[0] > 0 else ("worse" if g.mean() <= -BAND and ci[1] < 0 else ("no effect" if abs(g.mean()) < BAND else "inside CI"))
        head = "breaks headline" if (slope_d >= BAND and g.mean() >= BAND) else "-"
        rR, rI = recall_bins(df, r, a.frac); per = ", ".join(f"{s_}:{gg.mean():+.3f}" for s_, gg in g.groupby(g.index.str.split('/').str[0]))
        lines.append(f"{r:22s} gain(all) {g.mean():+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] {v:11s} | gain(4) {g4.mean():+.3f} | slope diff {slope_d:+.3f} {head} | F1 " + "/".join(f"{c[n].mean():.3f}" for n in (1, 2, 4, 8, 16, -1)) +
                     f" | P/R(all) {piv(df, r, a.frac, metric='precision@0.2')[-1].mean():.3f}/{piv(df, r, a.frac, metric='recall@0.2')[-1].mean():.3f} | recall range {np.array2string(rR, precision=3)} inc {np.array2string(rI, precision=3)} | {per}")
    # subset reference, same choices
    bs = piv(df, base, a.frac, ref="subset")
    lines.append(f"\n## subset ref, frac {a.frac}: gain at N=all, curve")
    for key, r in chosen.items():
        c = piv(df, r, a.frac, ref="subset")
        if c.empty or -1 not in c.columns:
            continue
        g = (c[-1] - bs[-1]).dropna(); ci = boot_ci(g)
        lines.append(f"{r:22s} gain(all) {g.mean():+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] | F1 " + "/".join(f"{c[n].mean():.3f}" for n in (1, 4, 16, -1)) + f" (base {bs[4].mean():.3f}/{bs[-1].mean():.3f} at 4/all)")
    # thresholded E108 (rate >= t) vs count at the same kept count
    if "thresh" in df.columns and df.thresh.notna().any():
        lines.append("\n## thresholded rates (val-chosen t) vs count at the same kept count, fixed ref, N=all")
        for r in [x for x in rankings if x.startswith("rate_geo_pred") or x.startswith("rate_smooth")]:
            best_t, best_v = None, -9
            for t in sorted(df.thresh.dropna().unique()):
                if dv is None:
                    break
                pv_ = piv(dv, r, None, thresh=t); pc_ = piv(dv, "count_match_" + r, None, thresh=t)
                if pv_.empty or -1 not in pv_.columns or pc_.empty:
                    continue
                val = (pv_[-1] - pc_[-1]).mean()
                if val > best_v:
                    best_t, best_v = t, val
            if best_t is None:
                continue
            pt = piv(df, r, None, thresh=best_t); pc = piv(df, "count_match_" + r, None, thresh=best_t)
            if pt.empty or -1 not in pt.columns:
                continue
            g = (pt[-1] - pc[-1]).dropna(); ci = boot_ci(g)
            kept = df[(df.ranking == r) & (df.ref == "fixed") & (df.N_req == -1) & np.isclose(df.thresh, best_t)].kept.mean()
            rR, _ = recall_bins(df, r, None, thresh=best_t); cR, _ = recall_bins(df, "count_match_" + r, None, thresh=best_t)
            lines.append(f"{r:22s} t={best_t}: kept {kept:.0f} voxels; F1 {pt[-1].mean():.3f} vs count-same-count {pc[-1].mean():.3f} | gain {g.mean():+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] | recall range {np.array2string(rR, precision=3)} vs {np.array2string(cR, precision=3)}")
    txt = "\n".join(lines); print(txt)
    (REPO / "results" / a.exp / a.mode / f"table_frac{a.frac}.txt").write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
