#!/usr/bin/env python3
"""Stage C (E20): train a ray-wise categorical depth posterior on the base backbone.

The backbone, attention stack and decoder are the base model's, initialised from
its released checkpoint; only the last convolution is replaced by a K-channel
one (`src/posterior_head.py`). Training data is the base model's own training
split (scene-disjoint, the same 12 scenes it used), so the held-out scenes used
for reconstruction stay unseen.

Loss is KL(soft target || predicted) over the depth bins. The soft target is a
Gaussian of `--label-width` bins around the true depth; a hard one-hot over 128
bins punishes a neighbouring bin as hard as the far side of the room, which
teaches the head to hedge for the wrong reason. An auxiliary expected-depth L1
is available but off by default and capped, because a large weight on it drives
the distribution back to a point.

    ECHO_DEPTH_ROOT=../hear360 REPLICA_ROOT=/root/storage/replica_0422 \
    python src/train_posterior.py --mode r2 --run posterior_r2 --gpu 0
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

CKPT = {"r2": "comparison/oaa_r2_fin/best.pth", "fb": "comparison/oaa_fb_fin/best.pth",
        "r6": "comparison/oaa_r6_fin/best.pth", "r8": "comparison/oaa_r8_fin/best.pth"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="r2", choices=sorted(CKPT))
    p.add_argument("--run", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--bins", type=int, default=128)
    p.add_argument("--d-min", type=float, default=0.1)
    p.add_argument("--d-max", type=float, default=10.0)
    p.add_argument("--label-width", type=float, default=1.0, help="soft label width in bins; 0 = one-hot")
    p.add_argument("--lambda-aux", type=float, default=0.0, help="weight of the auxiliary expected-depth L1 (<= 0.1)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup-epochs", type=float, default=2.0, help="head only, backbone frozen")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-train", type=int, default=0, help="debug: cap training batches per epoch")
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "posterior")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ.setdefault("REPLICA_ROOT", "/root/storage/replica_0422")
    os.environ.setdefault("R0422_SPLIT", "off3")
    os.environ.setdefault("DATA_MODULE", "data_0422")
    import torch
    import torch.nn.functional as F
    from posterior_head import PosteriorDepth, kl_loss, load_base, BASE_ROOT
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(BASE_ROOT / CKPT[a.mode], map_location="cpu", weights_only=False)
    base, poses, DM = load_base(ck["args"])
    base.load_state_dict(ck["state_dict"])
    model = PosteriorDepth(base, K=a.bins, d_min=a.d_min, d_max=a.d_max).to(dev)
    md = float(ck["args"].get("max_depth", 10.0))

    cwd = os.getcwd(); os.chdir(BASE_ROOT)
    try:
        tr = DM.loader("train", a.batch_size, True, a.num_workers, mode=a.mode)
        va = DM.loader("val", a.batch_size, False, a.num_workers, mode=a.mode)
    finally:
        os.chdir(cwd)
    out = a.out / a.run
    out.mkdir(parents=True, exist_ok=True)
    head_params = list(model.base.head.parameters())
    head_ids = {id(p) for p in head_params}
    rest = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": head_params, "lr": a.head_lr},
                             {"params": rest, "lr": a.lr}], weight_decay=1e-4)
    steps_per_epoch = max(1, (a.limit_train or len(tr)) // a.accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda")
    hist, best = [], float("inf")
    t0 = time.time()
    for ep in range(a.epochs):
        frozen = ep < a.warmup_epochs
        for p in rest:
            p.requires_grad_(not frozen)
        model.train()
        run_loss, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(tr):
            if a.limit_train and i >= a.limit_train:
                break
            spec = b["spec"].to(dev, non_blocking=True)
            depth = (b["depth"].to(dev) * md).squeeze(1)              # metres
            mask = b["mask"].to(dev).squeeze(1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(spec, view_poses=poses).float()
            tgt = model.soft_target(depth, a.label_width)
            loss = kl_loss(logits, tgt, mask)
            if a.lambda_aux > 0:
                ed = model.expected_depth(logits)
                loss = loss + a.lambda_aux * ((ed - depth).abs() * mask).sum() / mask.sum().clamp_min(1)
            scaler.scale(loss / a.accum).backward()
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            run_loss += float(loss); nb += 1
        model.eval()
        vl, vam, vem, vn = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for b in va:
                spec = b["spec"].to(dev); depth = (b["depth"].to(dev) * md).squeeze(1)
                mask = b["mask"].to(dev).squeeze(1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(spec, view_poses=poses).float()
                vl += float(kl_loss(logits, model.soft_target(depth, a.label_width), mask))
                am = (model.argmax_depth(logits) - depth).abs(); em = (model.expected_depth(logits) - depth).abs()
                vam += float((am * mask).sum() / mask.sum().clamp_min(1))
                vem += float((em * mask).sum() / mask.sum().clamp_min(1))
                vn += 1
        rec = dict(epoch=ep, frozen=bool(frozen), train_kl=run_loss / max(nb, 1),
                   val_kl=vl / max(vn, 1), val_argmax_mae=vam / max(vn, 1), val_expected_mae=vem / max(vn, 1),
                   lr=sched.get_last_lr()[0], minutes=(time.time() - t0) / 60)
        hist.append(rec)
        print(f"[{ep:02d}{'F' if frozen else ' '}] train KL {rec['train_kl']:.4f}  val KL {rec['val_kl']:.4f}  "
              f"argmax MAE {rec['val_argmax_mae']:.3f} m  expected MAE {rec['val_expected_mae']:.3f} m  "
              f"{rec['minutes']:.1f} min", flush=True)
        if rec["val_kl"] < best and not frozen:
            best = rec["val_kl"]
            torch.save({"state_dict": model.state_dict(), "args": vars(a) | {"base_args": ck["args"]},
                        "epoch": ep, "val": rec}, out / "best.pth")
        torch.save({"state_dict": model.state_dict(), "args": vars(a) | {"base_args": ck["args"]},
                    "epoch": ep, "val": rec}, out / "last.pth")
        (out / "history.json").write_text(json.dumps(hist, indent=1))
    (out / "config.yaml").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    (out / "git_commit.txt").write_text(
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout)
    print(f"done in {(time.time()-t0)/60:.1f} min, best val KL {best:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
