#!/usr/bin/env python3
"""Step 1: do the ground-truth ERP depths of a sequence fuse into one consistent 3D model?

For each candidate ERP convention the depths of consecutive steps are
unprojected with their poses and the nearest-neighbour distance from step i+1's
points to step i's is measured (0.15 m apart, the two views share most of the
room, so under the right convention the median distance is a few centimetres
and under a wrong one it is decimetres to metres). Also reports the floor
height spread as a second check (all floor points should sit at one y).

    python src/convention_check.py --scene apartment_2 --seq seq_0000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Sequence  # noqa: E402
from erp import CONVENTIONS, ray_dirs, unproject  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="apartment_2")
    ap.add_argument("--seq", default="seq_0000")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=12)
    a = ap.parse_args()
    S = Sequence(a.scene, a.seq)
    steps = S.steps[: a.pairs + 1]
    depths = [S.gt_depth(i) for i in steps]
    H, W = depths[0].shape
    print(f"{a.scene}/{a.seq}: {len(S)} steps, ERP {H}x{W}")
    print(f"{'convention':10s} {'median NN (m)':>14s} {'mean NN (m)':>12s} {'<5 cm':>7s} {'floor y std':>12s}")
    for name in CONVENTIONS:
        dirs = ray_dirs(H, W, name)
        clouds = [unproject(d, S.pose(i), dirs=dirs, stride=a.stride)[0] for d, i in zip(depths, steps)]
        nn = []
        for A, B in zip(clouds[:-1], clouds[1:]):
            dist, _ = cKDTree(A).query(B, k=1, workers=-1)
            nn.append(dist)
        nn = np.concatenate(nn)
        allp = np.concatenate(clouds)
        ys = allp[:, 1]
        floor = ys[ys < np.percentile(ys, 10)]
        print(f"{name:10s} {np.median(nn):14.3f} {nn.mean():12.3f} {100*(nn < 0.05).mean():6.1f}% {floor.std():12.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
