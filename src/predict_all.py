#!/usr/bin/env python3
"""Base-model predictions for every sequence of the test and val scenes with one
checkpoint load: outputs/pred/<name>/<scene>/<seq>.npz (same format as predict.py).
Used for the retrained front-ends of E100 (cause a).

    python src/predict_all.py --mode r2 --ckpt outputs/base_retrain/oaa_r2_w128_h32/best.pth --name r2_w128_h32 --gpu 0
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
from data import TEST_SCENES, VAL_SCENES, Sequence, sequences  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--ckpt", required=True, help="absolute path, or relative to the base checkout")
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--scenes", nargs="+", default=list(TEST_SCENES) + list(VAL_SCENES))
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "pred")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ.setdefault("REPLICA_ROOT", "/root/storage/replica_0422")
    os.environ.setdefault("R0422_SPLIT", "off3")
    import torch
    from predict import BASE_ROOT, CHAN, load_base_model, spec8
    ckpt = Path(a.ckpt) if os.path.isabs(a.ckpt) else (REPO / a.ckpt if (REPO / a.ckpt).exists() else BASE_ROOT / a.ckpt)
    ck_args = torch.load(ckpt, map_location="cpu", weights_only=False)["args"]
    win, hop = int(ck_args.get("stft_win", 400)), int(ck_args.get("stft_hop", 160))
    os.environ["STFT_WIN"] = str(win); os.environ["STFT_HOP"] = str(hop)
    device = torch.device("cuda")
    model, poses, args, DM = load_base_model(str(ckpt), device)
    chans = CHAN[a.mode]; assert len(poses) == len(chans)
    md = float(args.get("max_depth", 10.0)); window = int(DM.WINDOW)
    print(f"{ckpt} win {win} hop {hop} window {window} -> {a.out_dir / a.name}", flush=True)
    for sc in a.scenes:
        for sq in sequences(sc):
            out = a.out_dir / a.name / sc / f"{sq}.npz"
            if out.exists():
                continue
            S = Sequence(sc, sq); preds, steps = [], []; t0 = time.time()
            with torch.no_grad():
                for i in S.steps:
                    x = spec8(S.wav8(i, window)[chans], hop, device, win)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        D = model(x, view_poses=poses).float() * md
                    preds.append(D[0, 0].cpu().numpy().astype(np.float16)); steps.append(i)
            out.parent.mkdir(parents=True, exist_ok=True)
            meta = dict(scene=sc, seq=sq, mode=a.mode, n_obs=len(chans), ckpt=str(ckpt), base_root=str(BASE_ROOT), hop=hop, win=win,
                        max_depth=md, window=window, dropout_samples=0, seconds=round(time.time() - t0, 1), n_steps=len(steps))
            np.savez_compressed(out, pred=np.stack(preds), steps=np.array(steps), std=np.zeros((0,)), meta=json.dumps(meta))
            print(f"  {sc}/{sq} {len(steps)} steps {meta['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
