#!/usr/bin/env python3
"""E100 contribution table from cause_decomp.py outputs.

Reads results/E100_cause/<mode>[<tag>] (test) and <mode>[<tag>]_val (val, for the
hyper-parameter choice only) and prints, per CRITERIA.md:
  - the view-count curve (F1@0.2, precision, recall) per variant and reference
  - drop16 / dropAll per variant, recovery against base, paired bootstrap CI
  - the existence measurements (e)(b)(c)(d)

    python src/e100_table.py --mode r2
    python src/e100_table.py --mode r2 --tag _w128_h32   # the short-window model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from cause_decomp import pearson  # noqa: E402

BAND = 0.03


def curve(df, variant, ref, metric="f1@0.2", frac=1.0, seed=0):
    d = df[(df.variant == variant) & (df.ref == ref) & (df.frac == frac) & (df.seed == seed)]
    return d.pivot_table(index="seq_id", columns="N_req", values=metric)


def drops(df, variant, ref, frac=1.0, seed=0):
    """per-sequence drop16 = F1(4) - F1(16) and dropAll = F1(4) - F1(all)"""
    c = curve(df, variant, ref, frac=frac, seed=seed)
    if c.empty or 4 not in c.columns:
        return None
    out = pd.DataFrame({"drop16": c[4] - c[16]})
    out["dropAll"] = (c[4] - c[-1]) if -1 in c.columns else np.nan
    return out.dropna(subset=["drop16"])


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def verdict(rec, ci):
    if not np.isfinite(rec):
        return "[미확인]"
    if rec >= BAND and ci[0] > 0:
        return "meaningful"
    if abs(rec) < BAND:
        return "no effect"
    if rec <= -BAND and ci[1] < 0:
        return "worse"
    return "inside CI / below band"


def load(mode, tag, split):
    d = REPO / "results" / "E100_cause" / f"{mode}{tag}{'' if split == 'test' else '_val'}"
    f = d / "per_sequence.csv"
    if not f.exists():
        return None, None
    df = pd.read_csv(f)
    df["seq_id"] = df.scene + "/" + df.seq
    meas = json.loads((d / "measurements.json").read_text())
    return df, meas


def pool(meas, key):
    out = {}
    for m in meas:
        for k, v in m.get(key, {}).items():
            if not v:
                continue
            out[k] = {kk: out.get(k, {}).get(kk, 0) + vv for kk, vv in v.items()} if k in out else dict(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--tag", default="")
    ap.add_argument("--frac", type=float, default=1.0)
    a = ap.parse_args()
    df, meas = load(a.mode, a.tag, "test")
    if df is None:
        print(f"no test results for {a.mode}{a.tag}"); return 1
    dv, _ = load(a.mode, a.tag, "val")
    Ns = [1, 2, 4, 8, 16, -1]
    print(f"=== {a.mode}{a.tag}  test sequences {df.seq_id.nunique()}  frac {a.frac}  (val: {'yes' if dv is not None else 'none'})")

    # ---------- (e) reference growth and the base curve under both references
    print("\n## base curve, seed 0, mean over sequences (F1@0.2 | precision | recall)")
    for ref in ("subset", "fixed"):
        c = curve(df, "base", ref, frac=a.frac); p = curve(df, "base", ref, "precision@0.2", a.frac); r = curve(df, "base", ref, "recall@0.2", a.frac)
        print(f"{ref:7s} F1  " + " ".join(f"N={n:>3}:{c[n].mean():.3f}" for n in Ns if n in c))
        print(f"{'':7s} P   " + " ".join(f"N={n:>3}:{p[n].mean():.3f}" for n in Ns if n in p))
        print(f"{'':7s} R   " + " ".join(f"N={n:>3}:{r[n].mean():.3f}" for n in Ns if n in r))
    nref = df[(df.variant == "base") & (df.ref == "subset") & (df.frac == 1.0) & (df.seed == 0)].pivot_table(index="seq_id", columns="N_req", values="n_ref_subset")
    print(f"(e) n_ref(16)/n_ref(4) = {(nref[16] / nref[4]).mean():.2f}   n_ref(all)/n_ref(4) = {(nref[-1] / nref[4]).mean():.2f}")
    # per-scene std of drop16 (base) for the spread
    for ref in ("subset", "fixed"):
        d = drops(df, "base", ref, a.frac)
        d["scene"] = [s.split("/")[0] for s in d.index]
        print(f"(e) base drop16 [{ref}] mean {d.drop16.mean():+.3f}  sd over seq {d.drop16.std():.3f}  per scene " +
              ", ".join(f"{sc}:{g.drop16.mean():+.3f}±{g.drop16.std():.3f}" for sc, g in d.groupby('scene')) +
              f" | dropAll {d.dropAll.mean():+.3f}")
    d_sub = drops(df, "base", "subset", a.frac); d_fix = drops(df, "base", "fixed", a.frac)
    rec_e = (d_sub.drop16 - d_fix.drop16)
    ci = boot_ci(rec_e)
    print(f"(e) recovery = drop16(subset) - drop16(fixed) = {rec_e.mean():+.3f}  CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  -> {verdict(rec_e.mean(), ci)}")
    # seed spread
    for ref in ("subset", "fixed"):
        ds = [drops(df, "base", ref, a.frac, seed=s) for s in sorted(df.seed.unique())]
        print(f"    drop16 [{ref}] by subset seed: " + " ".join(f"{x.drop16.mean():+.3f}" for x in ds if x is not None))

    # ---------- variants: recovery under both references
    print("\n## recovery per variant (drop16(base) - drop16(variant), same sequences); curve at N=4/8/16/all")
    variants = [v for v in df.variant.unique() if v != "base"]
    sel = {}
    if dv is not None:
        # hyper-parameters chosen on val by drop16 recovery under the fixed reference
        for fam in ("smooth", "contra"):
            best, bv = None, -1e9
            for v in sorted(x for x in dv.variant.unique() if x.startswith(fam)):
                b = drops(dv, "base", "fixed", a.frac); x = drops(dv, v, "fixed", a.frac)
                j = b.join(x, lsuffix="_b", rsuffix="_v", how="inner")
                r = (j.drop16_b - j.drop16_v).mean()
                if r > bv:
                    best, bv = v, r
            sel[fam] = (best, bv)
        print("val-selected: " + ", ".join(f"{k}: {v[0]} (val recovery {v[1]:+.3f})" for k, v in sel.items()))
    rows = []
    for v in sorted(variants):
        for ref in ("subset", "fixed"):
            b = drops(df, "base", ref, a.frac); x = drops(df, v, ref, a.frac)
            if x is None or x.empty:
                continue
            j = b.join(x, lsuffix="_b", rsuffix="_v", how="inner")
            rec = j.drop16_b - j.drop16_v; ci = boot_ci(rec)
            recA = j.dropAll_b - j.dropAll_v
            c = curve(df, v, ref, frac=a.frac); cb = curve(df, "base", ref, frac=a.frac)
            g = lambda cc, n: cc[n].mean() if n in cc.columns else float("nan")
            flipped = (g(c, 16) >= g(c, 4) - 0.01) and (not (-1 in c.columns) or g(c, -1) >= g(c, 4) - 0.01)
            mark = " *" if any(v == s[0] for s in sel.values()) else ""
            rows.append(dict(variant=v + mark, ref=ref, n=len(j), recovery16=rec.mean(), ci_lo=ci[0], ci_hi=ci[1], recoveryAll=recA.mean(),
                             F1_4=g(c, 4), F1_16=g(c, 16), F1_all=g(c, -1), dF1_16=(c[16] - cb[16]).mean(),
                             dF1_all=((c[-1] - cb[-1]).mean() if -1 in c.columns else float("nan")),
                             flipped=flipped, verdict=verdict(rec.mean(), ci)))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))

    # ---------- (b) compact vs spread, and span
    print("\n## (b) spread (evenly spaced, seed 0) vs compact (consecutive) subsets, same N, F1@0.2")
    for ref in ("subset", "fixed"):
        cs = curve(df, "base", ref, frac=a.frac); cc = curve(df, "compact", ref, frac=a.frac)
        sp = df[(df.variant == "base") & (df.seed == 0) & (df.frac == 1.0) & (df.ref == ref)].pivot_table(index="seq_id", columns="N_req", values="span_m")
        spc = df[(df.variant == "compact") & (df.seed == 0) & (df.frac == 1.0) & (df.ref == ref)].pivot_table(index="seq_id", columns="N_req", values="span_m")
        for n in (4, 8, 16):
            if n in cc.columns:
                j = (cs[n] - cc[n]).dropna(); ci = boot_ci(j)
                print(f"[{ref}] N={n:2d}: spread {cs[n].mean():.3f} (span {sp[n].mean():.2f} m) - compact {cc[n].mean():.3f} (span {spc[n].mean():.2f} m) = {j.mean():+.3f} CI [{ci[0]:+.3f},{ci[1]:+.3f}] n={len(j)}")
    sp = pool(meas, "spread_precision")
    print("(b) precision@0.2 by viewing-angle spread tertile, within support bins (N=all, fixed ref):")
    for k, v in sp.items():
        pl, ph = v["right_low"] / max(v["n_low"], 1), v["right_high"] / max(v["n_high"], 1)
        print(f"    {k:10s} low-spread {pl:.3f} (n {v['n_low']})  high-spread {ph:.3f} (n {v['n_high']})  diff {ph - pl:+.3f}")

    # ---------- (c) correlations
    print("\n## (c) signed-error correlation (pooled Pearson r)")
    wv = pool(meas, "within_view")
    for sep in ("1-2deg", "5-10deg", "20-30deg"):
        s_ = {pl: pearson(wv.get(f'{sep}|{pl}|centred', {'n': 0})) for pl in ("same", "diff")}
        s_raw = {pl: pearson(wv.get(f'{sep}|{pl}|raw', {'n': 0})) for pl in ("same", "diff")}
        print(f"    within view {sep:8s}: same-plane r {s_['same']:.3f} (raw {s_raw['same']:.3f}, n {wv.get(f'{sep}|same|centred', {}).get('n', 0)}), "
              f"different-plane r {s_['diff']:.3f} (raw {s_raw['diff']:.3f}), gap {s_['same'] - s_['diff']:+.3f}")
    xv = pool(meas, "cross_view")
    for k in ("all", "<0.5m", "0.5-1.5m", ">=1.5m"):
        if f"{k}|centred" in xv:
            print(f"    cross view baseline {k:8s}: r {pearson(xv[f'{k}|centred']):.3f} (raw {pearson(xv[f'{k}|raw']):.3f}, n {xv[f'{k}|centred']['n']})")

    # ---------- (d) contradiction existence
    print("\n## (d) contradicted vs uncontradicted voxels (N=all, fixed reference, wrong = >0.2 m from reference)")
    tot = {}
    for m in meas:
        for k, v in m.get("contradiction", {}).items():
            if isinstance(v, int):
                tot[k] = tot.get(k, 0) + v
    if tot.get("n_vox"):
        wc = tot["wrong_contra"] / max(tot["n_contra"], 1); wu = tot["wrong_uncontra"] / max(tot["n_uncontra"], 1)
        print(f"    all voxels: contradicted {tot['n_contra'] / tot['n_vox']:.1%}, wrong rate contradicted {wc:.3f} vs uncontradicted {wu:.3f}, diff {wc - wu:+.3f}")
        if tot.get("n_mv"):
            wc = tot["wrong_mv_contra"] / max(tot["n_mv_contra"], 1); wu = tot["wrong_mv_uncontra"] / max(tot["n_mv"] - tot["n_mv_contra"], 1)
            print(f"    multi-view voxels (s>=2): {tot['n_mv'] / tot['n_vox']:.1%} of voxels, contradicted {tot['n_mv_contra'] / tot['n_mv']:.1%}, wrong rate contradicted {wc:.3f} vs uncontradicted {wu:.3f}, diff {wc - wu:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
