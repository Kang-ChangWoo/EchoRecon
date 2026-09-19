#!/usr/bin/env python3
"""Step 1b: which azimuth handedness is right? Fused ground truth against the floor plan.

The echo-localisation dataset renders the same Replica scenes with a wall-only
floor plan (map.png, 1 cm/px) and records the habitat -> plan frame offsets in
maps/<scene>/scene_meta.json. Points from the fused ground-truth depth at wall
height are projected onto the plan under each ERP convention; the convention
whose points land on the plan's obstacle pixels is the right one.

    python src/map_check.py --scene apartment_2 --seq seq_0000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Sequence  # noqa: E402
from erp import CONVENTIONS, ray_dirs, unproject  # noqa: E402

ECHOLOC = Path("/root/storage/echoloc_dataset/replica")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="apartment_2")
    ap.add_argument("--seq", default="seq_0000")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    meta = json.loads((ECHOLOC / "maps" / a.scene / "scene_meta.json").read_text())
    occ = np.array(Image.open(ECHOLOC / "maps" / a.scene / "map.png"))[:, :, 0]
    obstacle = occ != 255
    dist_px = distance_transform_edt(~obstacle)          # distance of every pixel to the nearest wall
    H, W = occ.shape
    cx = meta.get("world_cx_habitat"); cy = meta.get("world_cy_habitat_neg_z")
    if cx is None:
        # fall back to the keys as written in the file
        print("scene_meta keys:", list(meta)); return 1
    floor_y = meta.get("floor_y_habitat", None)
    S = Sequence(a.scene, a.seq)
    steps = S.steps[: a.steps]
    depths = [S.gt_depth(i, "radial") for i in steps]
    print(f"{a.scene}/{a.seq}: plan {H}x{W}, offsets cx={cx:.3f} cy={cy:.3f} floor_y={floor_y}")
    for name in CONVENTIONS:
        dirs = ray_dirs(*depths[0].shape, name)
        pts = np.concatenate([unproject(d, S.pose(i), dirs=dirs, stride=a.stride)[0] for d, i in zip(depths, steps)])
        # wall-height band: 0.6..1.8 m above the receiver's floor (receiver is 1.25 m up; y is up)
        yr = S.pose(steps[0])["position"][1]
        band = (pts[:, 1] > yr - 0.65) & (pts[:, 1] < yr + 0.55)
        P = pts[band]
        # habitat -> plan: x = hx - cx ; y = -hz - cy ; pixel col = x/0.01 + W/2, row = y/0.01 + H/2
        col = (P[:, 0] - cx) / 0.01 + W / 2
        row = (-P[:, 2] - cy) / 0.01 + H / 2
        inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        d = dist_px[row[inb].astype(int), col[inb].astype(int)] * 0.01
        print(f"{name:10s} points {inb.sum():7d}  median dist to wall {np.median(d):.3f} m  within 10 cm {100*(d < 0.10).mean():5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
