#!/usr/bin/env python3
"""Stage D (E30-E32, E40-E42): the comparison the project exists for.

One backbone, three ways of turning its output into a 3D model, over the
view-count curve:

  A point        the base model's point regression -> unproject -> voxels ranked
                 by cross-view support (the Stage A/B pipeline)
  B argmax       the posterior head's argmax depth -> the same pipeline, so the
                 only difference from A is which network produced the depth
  C posterior    the posterior head's full distribution -> soft surface
                 consensus, ranked by accumulated evidence

plus the oracle ranking as the ceiling. C is scored on two candidate sets,
because Stage B showed the choice decides the comparison: `C_on_B` uses exactly
the voxels B proposes, which isolates soft evidence from hard counting; `C_grid`
uses a voxel grid over the room, which lets the posterior support geometry no
argmax ever proposed. Both are reported.

The posterior's bins are per-face depth (what the base model was trained on), so
the query band is converted per ray; see `posterior.ViewPosterior`.

    python src/stage_d.py --run posterior_r2 --mode r2 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

NS = (1, 2, 4, 8, 16, 0)
FRACS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02)
# the room grid has ten to a hundred times more candidates than a point cloud,
# and its optimum sat on the 0.02 edge of the sweep; two smaller fractions
# bracket it
FRACS_GRID = FRACS + (0.01, 0.005)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="posterior run under outputs/posterior")
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--tau-band", type=float, default=0.15)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--max-cand", type=float, default=8e5)
    ap.add_argument("--convention", default="right0")
    ap.add_argument("--seq-limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="softmax temperature; 0 = the val-fitted value recorded by eval_posterior_rays")
    ap.add_argument("--tag", default="", help="suffix on the output directory, e.g. _T1")
    ap.add_argument("--rays-per-step", type=int, default=20000,
                    help="rays sampled per step for the confidence-versus-error (sparsification) record")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "E30_stage_d")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ.setdefault("REPLICA_ROOT", "/root/storage/replica_0422")
    os.environ.setdefault("R0422_SPLIT", "off3")
    os.environ.setdefault("DATA_MODULE", "data_0422")
    import torch
    import torch.nn.functional as F
    from data import TEST_SCENES, Sequence, sequences
    from erp import face_cos, ray_dirs
    from eval_fusion import resize_nearest, unproject_res
    from fuse import metrics, voxel_downsample
    from posterior import ViewPosterior, candidate_grid, depth_bins
    from posterior_head import BASE_ROOT, PosteriorDepth, load_base
    from stage_a import subset

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = REPO / "outputs" / "posterior" / a.run
    ck = torch.load(run / f"{a.ckpt}.pth", map_location="cpu", weights_only=False)
    args = ck["args"]
    base, poses, DM = load_base(args["base_args"])
    model = PosteriorDepth(base, K=args["bins"], d_min=args["d_min"], d_max=args["d_max"],
                           init_from_point_head=False).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    md = float(args["base_args"].get("max_depth", 10.0))
    window = int(DM.WINDOW)
    edges = np.linspace(args["d_min"], args["d_max"], args["bins"] + 1, dtype=np.float32)
    centres_t = model.centres
    T = a.temperature
    if T <= 0:
        rays_json = REPO / "results" / "E21_posterior_rays" / f"{a.run}_test.json"
        T = float(json.loads(rays_json.read_text()).get("temperature", 1.0)) if rays_json.exists() else 1.0
    print(f"softmax temperature {T:.3f}", flush=True)
    scenes_list = a.scenes or list(TEST_SCENES)
    out = a.out / f"{a.mode}{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    sp_conf, sp_err = [], []                      # per-ray confidence and argmax error, sampled
    rng = np.random.default_rng(0)
    t0 = time.time()

    # the posterior head consumes the same eight channels predict.py assembles
    from predict import CHAN, spec8
    chans = CHAN[a.mode]

    for sc in scenes_list:
        for si, sq in enumerate(sequences(sc)):
            if a.seq_limit and si >= a.seq_limit:
                break
            pf = REPO / "outputs" / "pred" / a.mode / sc / f"{sq}.npz"
            if not pf.exists():
                continue
            z = np.load(pf)
            P_point = z["pred"].astype(np.float32); steps = z["steps"].tolist()
            S = Sequence(sc, sq)
            H, W = P_point.shape[1:]
            dirs = ray_dirs(H, W, a.convention)
            fc = face_cos(dirs)
            # one forward pass per step: argmax depth (face) and the cdf over bins
            P_arg, cdfs, confs = [], [], []
            with torch.no_grad():
                for i in steps:
                    x = spec8(S.wav8(i, window)[chans], 160, dev)
                    lg = model(x, view_poses=poses).float() / T
                    p = F.softmax(lg, dim=1)[0]                       # (K, H, W)
                    arg = centres_t[lg[0].argmax(0)].cpu().numpy().astype(np.float32)
                    conf = p.max(0).values.cpu().numpy().astype(np.float32)   # per-ray confidence
                    P_arg.append(arg); confs.append(conf)
                    c = torch.zeros(args["bins"] + 1, H, W, device=dev)
                    c[1:] = p.cumsum(0)
                    cdfs.append(c.permute(1, 2, 0).reshape(-1, args["bins"] + 1).cpu().numpy())
                    g = resize_nearest(S.gt_depth(i, "face"), (H, W))
                    ok = np.isfinite(g) & (g > 0) & (g < a.max_depth)
                    pick = rng.choice(np.flatnonzero(ok), min(a.rays_per_step, int(ok.sum())), replace=False)
                    sp_conf.append(conf.ravel()[pick]); sp_err.append(np.abs(arg - g).ravel()[pick])
            for N in NS:
                idx = subset(len(steps), N, 0)
                refs, cl_A, cl_B, cf_B = [], [], [], []
                for j in idx:
                    pose = S.pose(steps[j])
                    cl_A.append(unproject_res(P_point[j], pose, dirs, a.max_depth, a.stride, "face")[0])
                    pb, vb = unproject_res(P_arg[j], pose, dirs, a.max_depth, a.stride, "face")[:2]
                    cl_B.append(pb); cf_B.append(confs[j][::a.stride, ::a.stride][vb])
                    g = resize_nearest(S.gt_depth(steps[j], "face"), (H, W))
                    refs.append(unproject_res(g, pose, dirs, a.max_depth, a.stride, "face")[0])
                ref, _ = voxel_downsample(np.concatenate(refs), a.voxel / 2)
                views = [ViewPosterior(cdfs[j], S.pose(steps[j])["position"], S.pose(steps[j])["rotation"],
                                       H, W, edges, space="face") for j in idx]
                base_row = dict(scene=sc, seq=sq, N_req=(N if N > 0 else -1), N=len(idx))

                def score_and_add(tag, pts, sc_, fracs=FRACS, extra=None):
                    order = np.argsort(-sc_)
                    for fr in fracs:
                        keep = order[: max(1, int(fr * len(pts)))]
                        m = metrics(pts[keep], ref, voxel=a.voxel)
                        rows.append({**base_row, "method": tag, "frac": fr,
                                     "n_cand": len(pts), "kept": len(keep), **(extra or {}), **m})

                vA, cA = voxel_downsample(np.concatenate(cl_A), a.voxel)
                score_and_add("A_point", vA, cA)
                vB, cB = voxel_downsample(np.concatenate(cl_B), a.voxel)
                score_and_add("B_argmax", vB, cB)
                evB = sum(v.band_mass(vB, a.tau_band) for v in views)
                score_and_add("C_on_B", vB, evB)
                # per-view confidence used before fusion: (D) the same voxels as B,
                # ranked by summed ray confidence instead of ray count; (E) rays below
                # a per-view confidence quantile dropped, then ranked by count
                allB, allC = np.concatenate(cl_B), np.concatenate(cf_B)
                vD, wD = voxel_downsample(allB, a.voxel, allC)
                score_and_add("D_conf_sum", vD, wD)
                for q in (0.5, 0.75):
                    keep_pts = [pb[cf >= np.quantile(cf, q)] for pb, cf in zip(cl_B, cf_B) if len(cf)]
                    vE, cE = voxel_downsample(np.concatenate(keep_pts), a.voxel)
                    score_and_add(f"E_conf_filter{int(q*100)}", vE, cE)
                lo = vB.min(0) - 0.3                      # box from the predictions only
                hi = vB.max(0) + 0.3
                n_est = np.prod((hi - lo) / a.voxel + 1)
                vx = a.voxel * max(1.0, (n_est / a.max_cand) ** (1 / 3))
                cand = candidate_grid(lo, hi, vx)
                evG = sum(v.band_mass(cand, a.tau_band) for v in views)
                score_and_add("C_grid", cand, evG, FRACS_GRID, {"grid_voxel": float(vx)})
                errB, _ = cKDTree(ref).query(vB, k=1, workers=-1)
                score_and_add("oracle", vB, -errB)
            print(f"[{sc}/{sq}] {len(steps)} steps, {len(rows)} rows, {(time.time()-t0)/60:.1f} min", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_sequence.csv", index=False)
    keys = ["method", "N_req", "frac"]
    num = [c for c in df.select_dtypes("number").columns if c not in keys]
    df.groupby(keys)[num].mean().to_csv(out / "aggregate.csv")
    (out / "config.yaml").write_text(json.dumps({k: str(v) for k, v in vars(a).items()} | {"temperature_used": T}, indent=1))
    (out / "git_commit.txt").write_text(
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout)
    # confidence versus error at the ray level: sparsification curve and its area
    # between the confidence ordering and the oracle (true-error) ordering
    cf = np.concatenate(sp_conf); er = np.concatenate(sp_err)
    fr = np.linspace(1.0, 0.05, 20)
    by_conf = np.argsort(-cf); by_err = np.argsort(er)
    curve_c = [float(er[by_conf[: max(1, int(f * len(er)))]].mean()) for f in fr]
    curve_o = [float(er[by_err[: max(1, int(f * len(er)))]].mean()) for f in fr]
    from scipy.stats import spearmanr
    rho = float(spearmanr(cf, er).correlation)
    dec = np.quantile(cf, np.linspace(0, 1, 11))
    mae_by_decile = [float(er[(cf >= dec[k]) & (cf <= dec[k + 1])].mean()) for k in range(10)]
    wrong_by_decile = [float((er[(cf >= dec[k]) & (cf <= dec[k + 1])] > 0.2).mean()) for k in range(10)]
    spars = dict(n_rays=int(len(er)), kept_frac=fr.tolist(), mae_by_confidence=curve_c, mae_by_oracle=curve_o,
                 ause=float(np.mean(np.array(curve_c) - np.array(curve_o))), spearman_conf_err=rho,
                 confidence_deciles=dec.tolist(), mae_by_decile=mae_by_decile, wrong_frac_by_decile=wrong_by_decile,
                 temperature=T)
    (out / "conf_sparsification.json").write_text(json.dumps(spars, indent=1))
    print(f"\nray confidence vs error ({len(er)/1e6:.1f} M rays, T={T:.3f}): spearman {rho:.3f}, AUSE {spars['ause']:.4f} m")
    print("  MAE by confidence decile (low->high): " + " ".join(f"{m:.3f}" for m in mae_by_decile))
    print("  wrong>0.2m by decile:                  " + " ".join(f"{w:.3f}" for w in wrong_by_decile))
    print(f"  keep top 50%% by conf: MAE {curve_c[10]:.3f}  (oracle {curve_o[10]:.3f}, all {curve_c[0]:.3f})".replace("%%", "%"))
    best = {}
    print(f"\nbest F1@0.2 over the kept-fraction sweep, by method and view count")
    hdr = sorted(df.N_req.unique(), key=lambda n: (n < 0, n))
    print(f"{'method':10s}" + "".join(f"{('all' if n < 0 else n):>8}" for n in hdr))
    for meth in ("A_point", "B_argmax", "C_on_B", "C_grid", "D_conf_sum", "E_conf_filter50", "E_conf_filter75", "oracle"):
        line = {}
        for n in hdr:
            s = df[(df.method == meth) & (df.N_req == n)]
            if len(s) == 0:
                continue
            g = s.groupby("frac")["f1@0.2"].mean()
            line[int(n)] = float(g.max())
        best[meth] = line
        print(f"{meth:10s}" + "".join(f"{line.get(int(n), float('nan')):8.3f}" for n in hdr))
    (out / "metrics.json").write_text(json.dumps(best, indent=1))
    print(f"\n{len(df)} rows in {(time.time()-t0)/60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
