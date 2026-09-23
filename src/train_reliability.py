#!/usr/bin/env python3
"""E111 trainer: reliability head on the released base model, head-only or joint.

    python src/train_reliability.py --mode r2 --variant headonly --gpu 0
    python src/train_reliability.py --mode r2 --variant joint --gpu 1
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
CKPT_ROOT = Path(os.environ.get("ECHORECON_CKPT", "/root/local1/changwoo/echorecon_ckpt"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="r2"); p.add_argument("--variant", choices=("headonly", "joint"), default="headonly")
    p.add_argument("--gpu", default="0"); p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--warmup-epochs", type=int, default=2); p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-5); p.add_argument("--head-lr", type=float, default=1e-3); p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4); p.add_argument("--accum", type=int, default=4); p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--limit-train", type=int, default=0)
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ.setdefault("REPLICA_ROOT", "/root/storage/replica_0422"); os.environ.setdefault("R0422_SPLIT", "off3"); os.environ.setdefault("DATA_MODULE", "data_0422")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from posterior_head import BASE_ROOT, load_base
    from reliability_head import ReliabilityDepth, bce_masked
    from train_posterior import CKPT
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"
    ck = torch.load(BASE_ROOT / CKPT[a.mode], map_location="cpu", weights_only=False)
    base, poses, DM = load_base(ck["args"]); base.load_state_dict(ck["state_dict"])
    model = ReliabilityDepth(base).to(dev); md = float(ck["args"].get("max_depth", 10.0))
    cwd = os.getcwd(); os.chdir(BASE_ROOT)
    try:
        tr = DM.loader("train", a.batch_size, True, a.num_workers, mode=a.mode); va = DM.loader("val", a.batch_size, False, a.num_workers, mode=a.mode)
    finally:
        os.chdir(cwd)
    run = f"reliability_{a.mode}_{a.variant}"; out = CKPT_ROOT / run; out.mkdir(parents=True, exist_ok=True)
    mirror = REPO / "outputs" / "reliability" / run; mirror.mkdir(parents=True, exist_ok=True)
    head_params = list(model.rel.parameters()); hid = {id(q) for q in head_params}; rest = [q for q in model.parameters() if id(q) not in hid]
    opt = torch.optim.AdamW([{"params": head_params, "lr": a.head_lr}, {"params": rest, "lr": a.lr}], weight_decay=1e-4)
    steps = max(1, (a.limit_train or len(tr)) // a.accum); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * steps)
    scaler = torch.amp.GradScaler("cuda")
    # positive rate of the target on the training rays (first 40 batches) for the class weight and the record
    with torch.no_grad():
        model.eval(); pos, n = 0.0, 0.0
        for i, b in enumerate(tr):
            if i >= 40:
                break
            spec = b["spec"].to(dev); depth = b["depth"].to(dev) * md; mask = b["mask"].to(dev).float()
            depth = depth if depth.dim() == 4 else depth.unsqueeze(1); mask = mask if mask.dim() == 4 else mask.unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d, _ = model(spec, view_poses=poses)
            tgt = ((d.float() * md - depth).abs() < a.tau).float(); pos += float((tgt * mask).sum()); n += float(mask.sum())
    p_train = pos / max(n, 1); pw = torch.tensor((1 - p_train) / max(p_train, 1e-3), device=dev)
    print(f"[{run}] train target positive rate {p_train:.3f} (pos_weight {float(pw):.2f}); train batches {len(tr)}", flush=True)
    hist, best, best_ep = [], float("inf"), -1; t0 = time.time()
    for ep in range(a.epochs):
        frozen = a.variant == "headonly" or ep < a.warmup_epochs
        for q in rest:
            q.requires_grad_(not frozen)
        model.train(); run_l, nb = 0.0, 0; opt.zero_grad(set_to_none=True)
        for i, b in enumerate(tr):
            if a.limit_train and i >= a.limit_train:
                break
            spec = b["spec"].to(dev, non_blocking=True); depth = b["depth"].to(dev) * md; mask = b["mask"].to(dev).float()
            depth = depth if depth.dim() == 4 else depth.unsqueeze(1); mask = mask if mask.dim() == 4 else mask.unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d, rl = model(spec, view_poses=poses)
            d = d.float() * md; rl = rl.float()
            tgt = ((d.detach() - depth).abs() < a.tau).float()
            loss = bce_masked(rl, tgt, mask, pw)
            if not frozen:
                loss = ((d - depth).abs() * mask).sum() / mask.sum().clamp_min(1) / md + a.lam * loss
            scaler.scale(loss / a.accum).backward()
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            run_l += float(loss.detach()); nb += 1
        model.eval(); vb, vn, vpos, vmae, vc, vauc_num = 0.0, 0.0, 0.0, 0.0, 0, []
        with torch.no_grad():
            for b in va:
                spec = b["spec"].to(dev); depth = b["depth"].to(dev) * md; mask = b["mask"].to(dev).float()
                depth = depth if depth.dim() == 4 else depth.unsqueeze(1); mask = mask if mask.dim() == 4 else mask.unsqueeze(1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    d, rl = model(spec, view_poses=poses)
                d = d.float() * md; rl = rl.float(); tgt = ((d - depth).abs() < a.tau).float()
                vb += float(bce_masked(rl, tgt, mask, torch.tensor(1.0, device=dev)) * mask.sum()); vn += float(mask.sum()); vpos += float((tgt * mask).sum())
                vmae += float(((d - depth).abs() * mask).sum())
                m_ = mask.bool(); vauc_num.append((rl[m_].flatten()[::97].cpu(), tgt[m_].flatten()[::97].cpu()))
        s_ = torch.cat([x for x, _ in vauc_num]).numpy(); y_ = torch.cat([y for _, y in vauc_num]).numpy()
        order = np.argsort(s_); r_ = np.empty(len(s_)); r_[order] = np.arange(1, len(s_) + 1); npos = y_.sum(); nneg = len(y_) - npos
        auc = float((r_[y_ == 1].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))
        rec = dict(epoch=ep, frozen=bool(frozen), train_loss=run_l / max(nb, 1), val_bce=vb / max(vn, 1), val_pos_rate=vpos / max(vn, 1), val_mae=vmae / max(vn, 1), val_auroc=auc, minutes=(time.time() - t0) / 60)
        hist.append(rec)
        print(f"[{ep:02d}{'F' if frozen else ' '}] train {rec['train_loss']:.4f}  val BCE {rec['val_bce']:.4f}  AUROC {auc:.3f}  pos {rec['val_pos_rate']:.3f}  MAE {rec['val_mae']:.3f}  {rec['minutes']:.1f} min", flush=True)
        state = {"state_dict": model.state_dict(), "args": vars(a) | {"base_args": ck["args"], "train_pos_rate": p_train}, "epoch": ep, "val": rec}
        if rec["val_bce"] < best:
            best, best_ep = rec["val_bce"], ep; torch.save(state, out / "best.pth")
        torch.save(state, out / "last.pth"); (mirror / "history.json").write_text(json.dumps(hist, indent=1))
        if a.patience and best_ep >= 0 and ep - best_ep >= a.patience:
            print(f"early stop at {ep} (best {best_ep})", flush=True); break
    done = {"best_val_bce": best, "best_epoch": best_ep, "epochs_run": len(hist), "train_pos_rate": p_train, "ckpt": str(out / "best.pth"), "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()}
    (mirror / "train_done.json").write_text(json.dumps(done, indent=1)); (out / "train_done.json").write_text(json.dumps(done, indent=1))
    print(f"done: {done}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
