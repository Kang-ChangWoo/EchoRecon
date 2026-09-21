#!/usr/bin/env python3
"""E104 prerequisite: per-ray wrong rate and MAE by azimuth (relative to the forward axis) and
elevation, pooled over the sequences of a split. Writes results/E104_lobe/<mode>_<split>.json
with the 12 x 6 (azimuth x elevation) wrong-rate table used as per-ray weights.

    python src/lobe_measure.py --mode r2 --split val
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from data import TEST_SCENES, VAL_SCENES, Sequence, sequences  # noqa: E402
from erp import ray_dirs, to_radial  # noqa: E402
from eval_fusion import resize_nearest  # noqa: E402

NA, NE = 12, 6
CFG = {}


def bins(H, W):
    """azimuth bin (0 = forward ±15°, increasing to the right) and elevation bin per pixel"""
    u = (np.arange(W) + 0.5) / W; theta = (u - 0.5) * 360.0          # degrees, centre column = forward
    az = np.floor(((theta + 15.0) % 360.0) / 30.0).astype(int) % NA
    v = (np.arange(H) + 0.5) / H; phi = (0.5 - v) * 180.0
    el = np.clip(np.floor((phi + 90.0) / 30.0).astype(int), 0, NE - 1)
    return np.broadcast_to(az[None, :], (H, W)), np.broadcast_to(el[:, None], (H, W))


def one(job):
    sc, sq = job; a = CFG
    z = np.load(Path(a["pred_dir"]) / sc / f"{sq}.npz"); P = z["pred"].astype(np.float32); steps = z["steps"].tolist()
    S = Sequence(sc, sq); H, W = P.shape[1:]; dirs = ray_dirs(H, W); az, el = bins(H, W)
    n = np.zeros((NA, NE)); wrong = np.zeros((NA, NE)); ae = np.zeros((NA, NE))
    for k, i in enumerate(steps):
        g = to_radial(resize_nearest(S.gt_depth(i, "face"), (H, W)), dirs, "face"); p = to_radial(P[k], dirs, "face")
        m = np.isfinite(g) & (g > 0) & (g < 10)
        e = np.abs(p - g)[m]; A = az[m]; E = el[m]
        idx = A * NE + E
        n += np.bincount(idx, minlength=NA * NE).reshape(NA, NE)
        wrong += np.bincount(idx, weights=(e > 0.2), minlength=NA * NE).reshape(NA, NE)
        ae += np.bincount(idx, weights=e, minlength=NA * NE).reshape(NA, NE)
    return n, wrong, ae


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="r2"); ap.add_argument("--split", default="val"); ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args(); a.pred_dir = a.pred_dir or str(REPO / "outputs" / "pred" / a.mode)
    CFG.update(vars(a))
    scenes = TEST_SCENES if a.split == "test" else VAL_SCENES
    jobs = [(sc, sq) for sc in scenes for sq in sequences(sc)]
    n = np.zeros((NA, NE)); wrong = np.zeros((NA, NE)); ae = np.zeros((NA, NE))
    with Pool(a.workers, initializer=lambda c: CFG.update(c), initargs=(dict(CFG),)) as pool:
        for n_, w_, e_ in pool.imap_unordered(one, jobs):
            n += n_; wrong += w_; ae += e_
    wr = wrong / np.maximum(n, 1); mae = ae / np.maximum(n, 1)
    wr_az = wrong.sum(1) / np.maximum(n.sum(1), 1); wr_el = wrong.sum(0) / np.maximum(n.sum(0), 1)
    out = REPO / "results" / "E104_lobe"; out.mkdir(parents=True, exist_ok=True)
    res = dict(mode=a.mode, split=a.split, n_seq=len(jobs), azimuth_deg_centres=[(30 * k) for k in range(NA)],
               elevation_deg_centres=[-75 + 30 * k for k in range(NE)], wrong_rate_az=wr_az.tolist(), wrong_rate_el=wr_el.tolist(),
               mae_az=(ae.sum(1) / np.maximum(n.sum(1), 1)).tolist(), mae_el=(ae.sum(0) / np.maximum(n.sum(0), 1)).tolist(),
               wrong_rate_az_el=wr.tolist(), mae_az_el=mae.tolist(), n_az_el=n.tolist())
    (out / f"{a.mode}_{a.split}.json").write_text(json.dumps(res, indent=1))
    print(f"{a.mode} {a.split}: wrong rate by azimuth (0=forward, +right): " + " ".join(f"{x:.2f}" for x in wr_az))
    print(f"   spread max-min {wr_az.max()-wr_az.min():.3f}; MAE by azimuth " + " ".join(f"{x:.2f}" for x in res['mae_az']))
    print(f"   wrong rate by elevation (-75..+75): " + " ".join(f"{x:.2f}" for x in wr_el) + f"; spread {wr_el.max()-wr_el.min():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
