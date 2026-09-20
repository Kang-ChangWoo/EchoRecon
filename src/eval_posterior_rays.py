#!/usr/bin/env python3
"""Stage C (E21-E25): does the posterior keep the true depth alive where the point is wrong?

Ray-level only, before any 3D fusion, on the held-out scenes of the base model's
own split. For each ray:

  argmax / expected depth error   the point the posterior would give
  NLL, Brier, ECE                 how well calibrated it is
  mass@±5/10/20/30 cm             probability inside a band around the truth
  top-K coverage                  is the true bin among the K most probable
  Wrong-Argmax Rescue@K           P(true bin in top K | argmax wrong by > tol)
  mode coverage / rescue@K        the same over the K most probable *local maxima*,
                                  asking whether any mode lands within tol of the
                                  truth. Bin-based top-K conflates a second mode
                                  with a wide single peak: the ten most probable
                                  bins of a unimodal Gaussian already span +-5
                                  bins, so only the mode version separates
                                  multimodality from peak width.
  n_modes                         local maxima above 10 % of the peak
  mass_given_wrong                probability within tol of the truth, on rays
                                  whose argmax is wrong

The last is the one the project turns on: when the point estimate is wrong, is
the right hypothesis still carried by the distribution? The fake Gaussian of
the same width is scored identically, so "the learned posterior is multimodal"
has to be earned against a blurred point (STOP CHECK B).

    python src/eval_posterior_rays.py --run posterior_r2 --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

KS = (1, 3, 5, 10)
BANDS = (0.05, 0.10, 0.20, 0.30)
SIGMAS = (0.05, 0.10, 0.20, 0.40)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="folder under outputs/posterior")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tol", type=float, default=0.2, help="an argmax further than this counts as wrong")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0, help="fitted on val, never on test")
    ap.add_argument("--fit-temperature", action="store_true", help="fit T on the val split and report it")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ.setdefault("REPLICA_ROOT", "/root/storage/replica_0422")
    os.environ.setdefault("R0422_SPLIT", "off3")
    os.environ.setdefault("DATA_MODULE", "data_0422")
    import torch
    import torch.nn.functional as F
    from posterior_head import BASE_ROOT, PosteriorDepth, load_base
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = REPO / "outputs" / "posterior" / a.run
    ck = torch.load(run / f"{a.ckpt}.pth", map_location="cpu", weights_only=False)
    args = ck["args"]
    base, poses, DM = load_base(args["base_args"])
    model = PosteriorDepth(base, K=args["bins"], d_min=args["d_min"], d_max=args["d_max"],
                           init_from_point_head=False).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    md = float(args["base_args"].get("max_depth", 10.0))
    cwd = os.getcwd(); os.chdir(BASE_ROOT)
    try:
        dl = DM.loader(a.split, a.batch_size, False, 4, mode=args["mode"])
    finally:
        os.chdir(cwd)

    def fit_T(loader):
        """One-parameter temperature on the validation split (never on test)."""
        # a single scalar needs only a sample of rays; keeping whole logit maps on
        # the card is several gigabytes, so rays are subsampled as they come
        logT = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([logT], max_iter=40)
        rng = torch.Generator().manual_seed(0)
        L, KK = [], []
        # every validation frame contributes a small sample of rays, so the scalar
        # is fitted across all three validation scenes rather than the first few
        # frames of one (the loader is not shuffled)
        per_batch = 1000
        with torch.no_grad():
            for i, b in enumerate(loader):
                lg = model(b["spec"].to(dev), view_poses=poses).float()
                d = (b["depth"].to(dev) * md).squeeze(1)
                msk = b["mask"].to(dev).squeeze(1).bool()
                lg = lg.permute(0, 2, 3, 1).reshape(-1, lg.shape[1])
                k = model.bin_of(d).reshape(-1); msk = msk.reshape(-1)
                idx = torch.nonzero(msk, as_tuple=False).squeeze(1)
                if idx.numel() > per_batch:
                    sel = torch.randperm(idx.numel(), generator=rng)[:per_batch].to(idx.device)
                    idx = idx[sel]
                L.append(lg[idx].cpu()); KK.append(k[idx].cpu())
                del lg
        L = torch.cat(L).to(dev); KK = torch.cat(KK).to(dev)

        def closure():
            opt.zero_grad()
            lp = F.log_softmax(L / logT.exp(), dim=1)
            loss = -lp.gather(1, KK.unsqueeze(1)).squeeze(1).mean()
            loss.backward(); return loss
        opt.step(closure)
        return float(logT.exp())

    if a.fit_temperature:
        cwd = os.getcwd(); os.chdir(BASE_ROOT)
        try:
            vdl = DM.loader("val", a.batch_size, False, 4, mode=args["mode"])
        finally:
            os.chdir(cwd)
        a.temperature = fit_T(vdl)
        print(f"temperature fitted on val: {a.temperature:.3f}")

    acc = {k: [] for k in ("argmax_mae", "expected_mae", "nll", "brier", "n_modes", "mass_given_wrong")}
    cov = {K: [] for K in KS}
    resc = {K: [] for K in KS}
    mcov = {K: [] for K in KS}
    mresc = {K: [] for K in KS}
    mass = {b: [] for b in BANDS}
    fake = {s: {"cov": {K: [] for K in KS}, "resc": {K: [] for K in KS},
                "mcov": {K: [] for K in KS}, "mresc": {K: [] for K in KS},
                "mass": {b: [] for b in BANDS}, "nll": [], "n_modes": [],
                "mass_given_wrong": []} for s in SIGMAS}

    def mode_mask(p):
        """True where a bin is a local maximum of the distribution along the bin axis."""
        left = torch.cat([p[:, :1] - 1, p[:, :-1]], 1)
        right = torch.cat([p[:, 1:], p[:, -1:] - 1], 1)
        return (p >= left) & (p >= right)

    def mode_stats(p, depth, m, wrong, centres, tol):
        """(coverage@K, rescue@K, mean number of modes) using local maxima only."""
        mm = mode_mask(p)
        pm = torch.where(mm, p, torch.full_like(p, -1.0))
        idx = pm.topk(max(KS), dim=1).indices                    # K most probable modes
        val = pm.gather(1, idx)
        near = ((centres[idx] - depth.unsqueeze(1)).abs() <= tol) & (val > 0)
        cov_k, res_k = {}, {}
        for K in KS:
            hit = near[:, :K].any(1)
            cov_k[K] = float(hit[m].float().mean())
            res_k[K] = float(hit[wrong].float().mean()) if wrong.any() else float("nan")
        n_modes = (mm & (p >= 0.1 * p.max(1, keepdim=True).values)).sum(1).float()
        return cov_k, res_k, float(n_modes[m].mean())
    conf_bins = np.zeros(15); conf_hit = np.zeros(15); conf_n = np.zeros(15)
    n_wrong = 0; n_tot = 0
    centres = model.centres
    with torch.no_grad():
        for i, b in enumerate(dl):
            if a.max_batches and i >= a.max_batches:
                break
            spec = b["spec"].to(dev); depth = (b["depth"].to(dev) * md).squeeze(1)
            m = b["mask"].to(dev).squeeze(1).bool()
            logits = model(spec, view_poses=poses).float() / a.temperature
            p = F.softmax(logits, dim=1)
            k_true = model.bin_of(depth)
            am = centres[logits.argmax(1)]; ex = (p * centres.view(1, -1, 1, 1)).sum(1)
            acc["argmax_mae"].append(float((am - depth).abs()[m].mean()))
            acc["expected_mae"].append(float((ex - depth).abs()[m].mean()))
            pt = p.gather(1, k_true.unsqueeze(1)).squeeze(1)
            acc["nll"].append(float((-pt.clamp_min(1e-12).log())[m].mean()))
            oh = F.one_hot(k_true, model.K).permute(0, 3, 1, 2).float()
            acc["brier"].append(float(((p - oh) ** 2).sum(1)[m].mean()))
            # reliability: confidence of the argmax against whether it is the true bin
            conf, pred = p.max(1)
            hit = (pred == k_true).float()
            cb = (conf.clamp(0, 0.999) * 15).long()
            for j in range(15):
                sel = m & (cb == j)
                if sel.any():
                    conf_n[j] += int(sel.sum()); conf_bins[j] += float(conf[sel].sum()); conf_hit[j] += float(hit[sel].sum())
            topk = p.topk(max(KS), dim=1).indices
            wrong = ((am - depth).abs() > a.tol) & m
            n_wrong += int(wrong.sum()); n_tot += int(m.sum())
            for K in KS:
                inK = (topk[:, :K] == k_true.unsqueeze(1)).any(1)
                cov[K].append(float(inK[m].float().mean()))
                if wrong.any():
                    resc[K].append(float(inK[wrong].float().mean()))
            ck, rk, nm = mode_stats(p, depth, m, wrong, centres, a.tol)
            for K in KS:
                mcov[K].append(ck[K])
                if wrong.any():
                    mresc[K].append(rk[K])
            acc["n_modes"].append(nm)
            lo_e, hi_e = model.edges[0], model.edges[-1]
            step = (hi_e - lo_e) / model.K
            cdf = torch.zeros(p.shape[0], model.K + 1, *p.shape[2:], device=dev)
            cdf[:, 1:] = p.cumsum(1)
            def cdf_at(u):
                # continuous bin coordinate: mass inside a bin is taken as uniform,
                # so band ends are interpolated instead of rounded to whole bins
                k = u.floor().clamp(0, model.K - 1).long()
                f = u - k.float()
                c0 = cdf.gather(1, k.unsqueeze(1)).squeeze(1)
                c1 = cdf.gather(1, (k + 1).unsqueeze(1)).squeeze(1)
                return c0 + f * (c1 - c0)
            for band in BANDS:
                u_lo = ((depth - band - lo_e) / step).clamp(0, model.K)
                u_hi = ((depth + band - lo_e) / step).clamp(0, model.K)
                mm = cdf_at(u_hi) - cdf_at(u_lo)
                mass[band].append(float(mm[m].mean()))
                if band == a.tol and wrong.any():
                    acc["mass_given_wrong"].append(float(mm[wrong].mean()))
            # the fake Gaussian control, on this model's own argmax depth
            for s in SIGMAS:
                dist = (centres.view(1, -1, 1, 1) - am.unsqueeze(1)) / s
                pf = torch.softmax(-0.5 * dist ** 2, dim=1)
                tf = pf.topk(max(KS), dim=1).indices
                for K in KS:
                    inK = (tf[:, :K] == k_true.unsqueeze(1)).any(1)
                    fake[s]["cov"][K].append(float(inK[m].float().mean()))
                    if wrong.any():
                        fake[s]["resc"][K].append(float(inK[wrong].float().mean()))
                ck, rk, nm = mode_stats(pf, depth, m, wrong, centres, a.tol)
                for K in KS:
                    fake[s]["mcov"][K].append(ck[K])
                    if wrong.any():
                        fake[s]["mresc"][K].append(rk[K])
                fake[s]["n_modes"].append(nm)
                ptf = pf.gather(1, k_true.unsqueeze(1)).squeeze(1)
                fake[s]["nll"].append(float((-ptf.clamp_min(1e-12).log())[m].mean()))
                cf = torch.zeros_like(cdf); cf[:, 1:] = pf.cumsum(1)
                for band in BANDS:
                    lo = ((depth - band - lo_e) / step).ceil().clamp(0, model.K).long()
                    hi = ((depth + band - lo_e) / step).floor().clamp(0, model.K).long()
                    mm = cf.gather(1, hi.unsqueeze(1)).squeeze(1) - cf.gather(1, lo.unsqueeze(1)).squeeze(1)
                    fake[s]["mass"][band].append(float(mm[m].mean()))
    ece = float(np.sum(conf_n / max(conf_n.sum(), 1) * np.abs(conf_bins / np.maximum(conf_n, 1) - conf_hit / np.maximum(conf_n, 1))))
    res = {"run": a.run, "split": a.split, "temperature": a.temperature, "tol": a.tol,
           "rays": n_tot, "wrong_frac": n_wrong / max(n_tot, 1), "ece": ece,
           **{k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()},
           "coverage": {str(K): float(np.mean(cov[K])) for K in KS},
           "rescue": {str(K): float(np.mean(resc[K])) if resc[K] else float("nan") for K in KS},
           "mode_coverage": {str(K): float(np.mean(mcov[K])) for K in KS},
           "mode_rescue": {str(K): float(np.mean(mresc[K])) if mresc[K] else float("nan") for K in KS},
           "mass": {str(b): float(np.mean(mass[b])) for b in BANDS},
           "fake": {str(s): {"coverage": {str(K): float(np.mean(fake[s]["cov"][K])) for K in KS},
                             "rescue": {str(K): float(np.mean(fake[s]["resc"][K])) if fake[s]["resc"][K] else float("nan") for K in KS},
                             "mode_coverage": {str(K): float(np.mean(fake[s]["mcov"][K])) for K in KS},
                             "mode_rescue": {str(K): float(np.mean(fake[s]["mresc"][K])) if fake[s]["mresc"][K] else float("nan") for K in KS},
                             "n_modes": float(np.mean(fake[s]["n_modes"])),
                             "mass": {str(b): float(np.mean(fake[s]["mass"][b])) for b in BANDS},
                             "nll": float(np.mean(fake[s]["nll"]))} for s in SIGMAS}}
    out = a.out or REPO / "results" / "E21_posterior_rays" / f"{a.run}_{a.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"\n{a.run} on {a.split}: {n_tot/1e6:.1f} M rays, {100*res['wrong_frac']:.1f} % of argmaxes wrong by > {a.tol} m")
    print(f"  argmax MAE {res['argmax_mae']:.3f} m   expected MAE {res['expected_mae']:.3f} m   "
          f"NLL {res['nll']:.3f}   Brier {res['brier']:.4f}   ECE {ece:.4f}")
    print(f"  top-K coverage " + "  ".join(f"K={K}: {100*res['coverage'][str(K)]:.1f}%" for K in KS))
    print(f"  rescue@K       " + "  ".join(f"K={K}: {100*res['rescue'][str(K)]:.1f}%" for K in KS))
    print(f"  mode cov@K     " + "  ".join(f"K={K}: {100*res['mode_coverage'][str(K)]:.1f}%" for K in KS))
    print(f"  mode rescue@K  " + "  ".join(f"K={K}: {100*res['mode_rescue'][str(K)]:.1f}%" for K in KS)
          + f"   modes/ray {res['n_modes']:.2f}   mass at truth when argmax wrong {100*res['mass_given_wrong']:.1f}%")
    print(f"  mass in band   " + "  ".join(f"±{int(100*b)}cm: {100*res['mass'][str(b)]:.1f}%" for b in BANDS))
    print("  fake Gaussian of the same argmax:")
    for s in SIGMAS:
        f = res["fake"][str(s)]
        print(f"    sigma {s:.2f}: rescue " + " ".join(f"K={K}:{100*f['rescue'][str(K)]:.1f}%" for K in KS)
              + f" | mode rescue " + " ".join(f"K={K}:{100*f['mode_rescue'][str(K)]:.1f}%" for K in KS)
              + f" | modes/ray {f['n_modes']:.2f}  mass ±20cm {100*f['mass']['0.2']:.1f}%  NLL {f['nll']:.3f}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
