#!/usr/bin/env python3
"""Base-model ERP depth for every step of a sequence -> outputs/pred/<mode>/<scene>/<seq>.npz

The base model is imported from its sibling checkout (ECHO_DEPTH_ROOT, default
../hear360 relative to this repository) and run as its own sequence-inference
script does. --mode picks the observation set and the matching
checkpoint: r2 (default) one heading, the front binaural pair [000 L,R]; fb two
headings, front and back [000 L,R | 180 L,R]; r6 three; r8 all four. Magnitude STFT (n_fft 512,
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

# which of the eight channels of a step each observation set uses. The step's
# channels are [000 L,R | 090 L,R | 180 L,R | 270 L,R]; a "heading" contributes
# its two ears together.
CHAN = {"r2": [0, 1],                    # one heading (front binaural pair)
        "fb": [0, 1, 4, 5],              # two headings, front and back
        "r6": [0, 1, 2, 3, 6, 7],        # three headings, front / +90 / +270
        "r8": [0, 1, 2, 3, 4, 5, 6, 7]}  # four headings
CKPT = {m: f"comparison/oaa_{m}_fin/best.pth" for m in CHAN}


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


def spec8(w8: np.ndarray, hop: int, device, win: int = 400):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(w8).to(device)
    s = torch.stft(t, n_fft=512, win_length=win, hop_length=hop, window=torch.hann_window(win, device=device),
                   center=True, return_complex=True).abs()[:, :256]
    return F.interpolate(s.unsqueeze(0), size=(256, 512), mode="nearest")      # (1, 8, 256, 512)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--mode", default="r2", choices=sorted(CHAN),
                    help="observation set: r2 one heading (2 ch), fb two headings front/back (4 ch), "
                         "r6 three headings (6 ch), r8 four headings (8 ch)")
    ap.add_argument("--ckpt", default=None, help="relative to ECHO_DEPTH_ROOT; default the base model's final run for --mode")
    ap.add_argument("--hop", type=int, default=None, help="STFT hop; default the checkpoint's (released 160)")
    ap.add_argument("--win", type=int, default=None, help="STFT window; default the checkpoint's (released 400)")
    ap.add_argument("--name", default=None, help="output set name under --out-dir (default --mode)")
    ap.add_argument("--dropout-samples", type=int, default=0)
    ap.add_argument("--dropout-kmax", type=int, default=None, help="at most this many observations zeroed per sample (default half)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "pred")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.ckpt is None:
        a.ckpt = CKPT[a.mode]
    # retrained front-ends (E100 cause a) record their STFT window / hop in the checkpoint;
    # the base data module reads them from the environment at import, so set them first
    ck_args = torch.load(a.ckpt if os.path.isabs(a.ckpt) else BASE_ROOT / a.ckpt, map_location="cpu", weights_only=False)["args"]
    a.win = a.win or int(ck_args.get("stft_win", 400)); a.hop = a.hop or int(ck_args.get("stft_hop", 160))
    os.environ["STFT_WIN"] = str(a.win); os.environ["STFT_HOP"] = str(a.hop)
    a.name = a.name or a.mode
    chans = CHAN[a.mode]; nch = len(chans)
    if a.dropout_kmax is None:
        a.dropout_kmax = nch // 2
    model, poses, args, DM = load_base_model(a.ckpt, device)
    assert len(poses) == nch, f"checkpoint has {len(poses)} observations, --mode {a.mode} needs {nch}"
    md = float(args.get("max_depth", 10.0))
    window = int(DM.WINDOW)
    S = Sequence(a.scene, a.seq)
    rng = np.random.default_rng(a.seed)
    preds, stds, steps = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in S.steps:
            x = spec8(S.wav8(i, window)[chans], a.hop, device, a.win)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                D = model(x, view_poses=poses).float() * md
            preds.append(D[0, 0].cpu().numpy().astype(np.float16)); steps.append(i)
            if a.dropout_samples:
                runs = []
                for _ in range(a.dropout_samples):
                    k = int(rng.integers(1, a.dropout_kmax + 1))
                    drop = rng.choice(nch, size=k, replace=False)
                    xd = x.clone(); xd[:, drop] = 0
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        runs.append((model(xd, view_poses=poses).float() * md)[0, 0].cpu().numpy())
                stds.append(np.std(np.stack(runs), 0).astype(np.float16))
    out = a.out_dir / a.name / a.scene / f"{a.seq}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(scene=a.scene, seq=a.seq, mode=a.mode, n_obs=nch, ckpt=str(a.ckpt), base_root=str(BASE_ROOT), hop=a.hop, win=a.win, max_depth=md,
                window=window, dropout_samples=a.dropout_samples, dropout_kmax=a.dropout_kmax, seed=a.seed,
                seconds=round(time.time() - t0, 1), n_steps=len(steps))
    np.savez_compressed(out, pred=np.stack(preds), steps=np.array(steps),
                        std=(np.stack(stds) if stds else np.zeros((0,))), meta=json.dumps(meta))
    print(f"wrote {out}  {len(steps)} steps, mode {a.mode}, hop {a.hop}, {meta['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
