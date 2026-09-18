#!/usr/bin/env python3
"""Base-model ERP depth for every step of a sequence -> outputs/pred/<scene>/<seq>.npz

The base model is imported from its sibling checkout (ECHO_DEPTH_ROOT, default
../hear360 relative to this repository) and run exactly as its own
sequence-inference script does: the eight binaural channels of a step in the
order [000 L,R | 090 L,R | 180 L,R | 270 L,R], magnitude STFT (n_fft 512,
win 400, hop --hop), first 256 bins, nearest-resized to 256x512, view poses
from the checkpoint's mode. The hop is sweepable; the trained recipe is 160 and
every output records the hop it was produced with.

With --dropout-samples K the model is also run K times with a random subset of
observations zeroed (the base model's own training-time observation dropout),
and the per-pixel standard deviation across the K runs is stored as a candidate
certainty signal (step 3 of the plan).

    ECHO_DEPTH_ROOT=../hear360 python src/predict.py --scene apartment_2 --seq seq_0000 --gpu 0
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
BASE_ROOT = Path(os.environ.get("ECHO_DEPTH_ROOT", REPO.parent / "hear360")).resolve()
sys.path.insert(0, str(HERE))
from data import Sequence  # noqa: E402


def load_base_model(ckpt_rel: str, device):
    os.environ.setdefault("DATA_MODULE", "data_0422")
    sys.path.insert(0, str(BASE_ROOT))
    cwd = os.getcwd()
    os.chdir(BASE_ROOT)                      # its checkpoint paths are relative to its root
    try:
        import torch
        from core.ckpt import build
        from core.data import get_data_module
        DM = get_data_module()
        ck = torch.load(ckpt_rel, map_location="cpu", weights_only=False)
        model, dmode, nch, kind, poses = build(ck["args"], DM)
        model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    finally:
        os.chdir(cwd)
    return model, poses, ck["args"], DM


def spec8(w8: np.ndarray, hop: int, device):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(w8).to(device)
    s = torch.stft(t, n_fft=512, win_length=400, hop_length=hop, window=torch.hann_window(400, device=device),
                   center=True, return_complex=True).abs()[:, :256]
    return F.interpolate(s.unsqueeze(0), size=(256, 512), mode="nearest")      # (1, 8, 256, 512)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--ckpt", default="comparison/oaa_r8_fin/best.pth", help="relative to ECHO_DEPTH_ROOT")
    ap.add_argument("--hop", type=int, default=160)
    ap.add_argument("--dropout-samples", type=int, default=0)
    ap.add_argument("--dropout-kmax", type=int, default=4, help="at most this many of the 8 observations zeroed per sample")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "pred")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, poses, args, DM = load_base_model(a.ckpt, device)
    md = float(args.get("max_depth", 10.0))
    window = int(DM.WINDOW)
    S = Sequence(a.scene, a.seq)
    rng = np.random.default_rng(a.seed)
    preds, stds, steps = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in S.steps:
            x = spec8(S.wav8(i, window), a.hop, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                D = model(x, view_poses=poses).float() * md
            preds.append(D[0, 0].cpu().numpy().astype(np.float16)); steps.append(i)
            if a.dropout_samples:
                runs = []
                for _ in range(a.dropout_samples):
                    k = int(rng.integers(1, a.dropout_kmax + 1))
                    drop = rng.choice(8, size=k, replace=False)
                    xd = x.clone(); xd[:, drop] = 0
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        runs.append((model(xd, view_poses=poses).float() * md)[0, 0].cpu().numpy())
                stds.append(np.std(np.stack(runs), 0).astype(np.float16))
    out = a.out_dir / a.scene / f"{a.seq}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(scene=a.scene, seq=a.seq, ckpt=a.ckpt, base_root=str(BASE_ROOT), hop=a.hop, max_depth=md,
                window=window, dropout_samples=a.dropout_samples, dropout_kmax=a.dropout_kmax, seed=a.seed,
                seconds=round(time.time() - t0, 1), n_steps=len(steps))
    np.savez_compressed(out, pred=np.stack(preds), steps=np.array(steps),
                        std=(np.stack(stds) if stds else np.zeros((0,))), meta=json.dumps(meta))
    print(f"wrote {out}  {len(steps)} steps, hop {a.hop}, {meta['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
