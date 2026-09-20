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
    scenes_list = a.scenes or list(TEST_SCENES)
    out = a.out / a.mode
    out.mkdir(parents=True, exist_ok=True)
    rows = []
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
            P_arg, cdfs = [], []
            with torch.no_grad():
                for i in steps:
                    x = spec8(S.wav8(i, window)[chans], 160, dev)
                    lg = model(x, view_poses=poses).float()
                    p = F.softmax(lg, dim=1)[0]                       # (K, H, W)
                    P_arg.append(centres_t[lg[0].argmax(0)].cpu().numpy().astype(np.float32))
                    c = torch.zeros(args["bins"] + 1, H, W, device=dev)
                    c[1:] = p.cumsum(0)
                    cdfs.append(c.permute(1, 2, 0).reshape(-1, args["bins"] + 1).cpu().numpy())
            for N in NS:
                idx = subset(len(steps), N, 0)
                refs, cl_A, cl_B = [], [], []
                for j in idx:
                    pose = S.pose(steps[j])
                    cl_A.append(unproject_res(P_point[j], pose, dirs, a.max_depth, a.stride, "face")[0])
                    cl_B.append(unproject_res(P_arg[j], pose, dirs, a.max_depth, a.stride, "face")[0])
                    g = resize_nearest(S.gt_depth(steps[j], "face"), (H, W))
                    refs.append(unproject_res(g, pose, dirs, a.max_depth, a.stride, "face")[0])
                ref, _ = voxel_downsample(np.concatenate(refs), a.voxel / 2)
                views = [ViewPosterior(cdfs[j], S.pose(steps[j])["position"], S.pose(steps[j])["rotation"],
                                       H, W, edges, space="face") for j in idx]
                base_row = dict(scene=sc, seq=sq, N_req=(N if N > 0 else -1), N=len(idx))

                def score_and_add(tag, pts, sc_):
                    order = np.argsort(-sc_)
                    for fr in FRACS:
                        keep = order[: max(1, int(fr * len(pts)))]
                        m = metrics(pts[keep], ref, voxel=a.voxel)
                        rows.append({**base_row, "method": tag, "frac": fr,
                                     "n_cand": len(pts), "kept": len(keep), **m})

                vA, cA = voxel_downsample(np.concatenate(cl_A), a.voxel)
                score_and_add("A_point", vA, cA)
                vB, cB = voxel_downsample(np.concatenate(cl_B), a.voxel)
                score_and_add("B_argmax", vB, cB)
                evB = sum(v.band_mass(vB, a.tau_band) for v in views)
                score_and_add("C_on_B", vB, evB)
                lo = np.minimum(ref.min(0), vB.min(0)) - 0.2
                hi = np.maximum(ref.max(0), vB.max(0)) + 0.2
                n_est = np.prod((hi - lo) / a.voxel + 1)
                vx = a.voxel * max(1.0, (n_est / a.max_cand) ** (1 / 3))
                cand = candidate_grid(lo, hi, vx)
                evG = sum(v.band_mass(cand, a.tau_band) for v in views)
                score_and_add("C_grid", cand, evG)
                errB, _ = cKDTree(ref).query(vB, k=1, workers=-1)
                score_and_add("oracle", vB, -errB)
            print(f"[{sc}/{sq}] {len(steps)} steps, {len(rows)} rows, {(time.time()-t0)/60:.1f} min", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_sequence.csv", index=False)
    keys = ["method", "N_req", "frac"]
    num = [c for c in df.select_dtypes("number").columns if c not in keys]
    df.groupby(keys)[num].mean().to_csv(out / "aggregate.csv")
    (out / "config.yaml").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    (out / "git_commit.txt").write_text(
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout)
    best = {}
    print(f"\nbest F1@0.2 over the kept-fraction sweep, by method and view count")
    hdr = sorted(df.N_req.unique(), key=lambda n: (n < 0, n))
    print(f"{'method':10s}" + "".join(f"{('all' if n < 0 else n):>8}" for n in hdr))
    for meth in ("A_point", "B_argmax", "C_on_B", "C_grid", "oracle"):
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
