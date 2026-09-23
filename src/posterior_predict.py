#!/usr/bin/env python3
"""Posterior-head argmax depth and per-ray confidence for every step of the test and val
sequences, in the prediction-file format: outputs/pred/<name>/<scene>/<seq>.npz with
pred (argmax face depth), conf (max softmax at the val-fitted temperature), steps, meta.

    python src/posterior_predict.py --run posterior_r2 --mode r2 --gpu 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--name", default=None, help="output set name (default <mode>_post)")
    ap.add_argument("--hop", type=int, default=160)
    ap.add_argument("--topk", type=int, default=0, help="also store the k highest-mass local modes per ray (bin index, mass) for E110")
    ap.add_argument("--split", default="heldout", choices=("heldout", "train", "all"), help="heldout = test + val scenes (default); train = the 12 training scenes")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ.setdefault("REPLICA_ROOT", "/root/storage/replica_0422")
    os.environ.setdefault("R0422_SPLIT", "off3")
    os.environ.setdefault("DATA_MODULE", "data_0422")
    import torch
    import torch.nn.functional as F
    from data import TEST_SCENES, VAL_SCENES, Sequence, sequences, scenes as all_scenes
    from posterior_head import PosteriorDepth, load_base
    from predict import CHAN, spec8
    dev = "cuda"
    run = REPO / "outputs" / "posterior" / a.run
    ck = torch.load(run / "best.pth", map_location="cpu", weights_only=False); args = ck["args"]
    base, poses, DM = load_base(args["base_args"])
    model = PosteriorDepth(base, K=args["bins"], d_min=args["d_min"], d_max=args["d_max"], init_from_point_head=False).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    rays_json = REPO / "results" / "E21_posterior_rays" / f"{a.run}_test.json"
    T = float(json.loads(rays_json.read_text())["temperature"])
    window = int(DM.WINDOW); chans = CHAN[a.mode]; centres = model.centres
    name = a.name or f"{a.mode}_post"
    print(f"{a.run}: T={T:.3f}, K={args['bins']}, window {window} -> outputs/pred/{name}", flush=True)
    held = list(TEST_SCENES) + list(VAL_SCENES)
    train = [x for x in all_scenes() if x not in held]
    for sc in {"heldout": held, "train": train, "all": held + train}[a.split]:
        for sq in sequences(sc):
            out = REPO / "outputs" / "pred" / name / sc / f"{sq}.npz"
            if out.exists():
                continue
            S = Sequence(sc, sq); preds, confs, steps = [], [], []; t0 = time.time(); tk_idx, tk_mass = [], []
            with torch.no_grad():
                for i in S.steps:
                    x = spec8(S.wav8(i, window)[chans], a.hop, dev)
                    lg = model(x, view_poses=poses).float() / T
                    p = F.softmax(lg, dim=1)[0]
                    preds.append(centres[lg[0].argmax(0)].cpu().numpy().astype(np.float16))
                    confs.append(p.max(0).values.cpu().numpy().astype(np.float16)); steps.append(i)
                    if a.topk:
                        # local modes: bins whose mass exceeds both neighbours; mass of a mode = its own
                        # bin plus the two neighbours (a 3-bin window, 0.23 m), so k modes ~ k candidates
                        pm = torch.nn.functional.max_pool1d(p.permute(1, 2, 0).reshape(-1, 1, p.shape[0]), 3, 1, 1).reshape(p.shape[1], p.shape[2], p.shape[0]).permute(2, 0, 1)
                        w3 = torch.nn.functional.avg_pool1d(p.permute(1, 2, 0).reshape(-1, 1, p.shape[0]), 3, 1, 1).reshape(p.shape[1], p.shape[2], p.shape[0]).permute(2, 0, 1) * 3
                        is_mode = (p >= pm)
                        score = torch.where(is_mode, w3, torch.zeros_like(w3))
                        v_, i_ = score.topk(a.topk, dim=0)
                        tk_idx.append(i_.cpu().numpy().astype(np.uint8)); tk_mass.append(v_.cpu().numpy().astype(np.float16))
            out.parent.mkdir(parents=True, exist_ok=True)
            meta = dict(scene=sc, seq=sq, mode=a.mode, run=a.run, temperature=T, hop=a.hop, n_steps=len(steps), seconds=round(time.time() - t0, 1))
            extra = {"topk_idx": np.stack(tk_idx), "topk_mass": np.stack(tk_mass), "centres": centres.cpu().numpy()} if a.topk else {}
            np.savez_compressed(out, pred=np.stack(preds), conf=np.stack(confs), steps=np.array(steps), std=np.zeros((0,)), meta=json.dumps(meta), **extra)
            print(f"  {sc}/{sq} {len(steps)} steps {meta['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
