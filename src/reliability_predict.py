#!/usr/bin/env python3
"""E111: depth and per-ray reliability of a ReliabilityDepth checkpoint for every sequence of a
split, in the prediction-file format (pred = face depth, conf = reliability in [0,1]).

    python src/reliability_predict.py --run reliability_r2_headonly --mode r2 --gpu 0 --split heldout
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
CKPT_ROOT = Path(os.environ.get("ECHORECON_CKPT", "/root/local1/changwoo/echorecon_ckpt"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--mode", default="r2"); ap.add_argument("--gpu", default="0")
    ap.add_argument("--split", default="heldout", choices=("heldout", "train", "all")); ap.add_argument("--name", default=None); ap.add_argument("--hop", type=int, default=160)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ.setdefault("REPLICA_ROOT", "/root/storage/replica_0422"); os.environ.setdefault("R0422_SPLIT", "off3"); os.environ.setdefault("DATA_MODULE", "data_0422")
    import torch
    from data import TEST_SCENES, VAL_SCENES, Sequence, sequences, scenes as all_scenes
    from posterior_head import load_base
    from predict import CHAN, spec8
    from reliability_head import ReliabilityDepth
    ck = torch.load(CKPT_ROOT / a.run / "best.pth", map_location="cpu", weights_only=False); args = ck["args"]
    base, poses, DM = load_base(args["base_args"]); model = ReliabilityDepth(base).cuda(); model.load_state_dict(ck["state_dict"]); model.eval()
    md = float(args["base_args"].get("max_depth", 10.0)); window = int(DM.WINDOW); chans = CHAN[a.mode]; name = a.name or a.run
    held = list(TEST_SCENES) + list(VAL_SCENES); train = [x for x in all_scenes() if x not in held]
    for sc in {"heldout": held, "train": train, "all": held + train}[a.split]:
        for sq in sequences(sc):
            out = REPO / "outputs" / "pred" / name / sc / f"{sq}.npz"
            if out.exists():
                continue
            S = Sequence(sc, sq); preds, rels, steps = [], [], []; t0 = time.time()
            with torch.no_grad():
                for i in S.steps:
                    x = spec8(S.wav8(i, window)[chans], a.hop, "cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        d, rl = model(x, view_poses=poses)
                    preds.append((d.float()[0, 0] * md).cpu().numpy().astype(np.float16)); rels.append(torch.sigmoid(rl.float())[0, 0].cpu().numpy().astype(np.float16)); steps.append(i)
            out.parent.mkdir(parents=True, exist_ok=True)
            meta = dict(scene=sc, seq=sq, mode=a.mode, run=a.run, epoch=int(ck["epoch"]), hop=a.hop, n_steps=len(steps), seconds=round(time.time() - t0, 1))
            np.savez_compressed(out, pred=np.stack(preds), conf=np.stack(rels), steps=np.array(steps), std=np.zeros((0,)), meta=json.dumps(meta))
            print(f"  {sc}/{sq} {len(steps)} steps {meta['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
