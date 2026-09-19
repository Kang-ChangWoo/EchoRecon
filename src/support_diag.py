#!/usr/bin/env python3
"""Step 3: refine the support signal — three-state agreement, restricted by baseline.

Step 1 showed that counting how many views put a point in a voxel already beats
the other cheap scores, and step 2 showed the errors are fine-grained and
view-dependent, so views that disagree should be informative. Counting points
cannot express disagreement: a view that saw straight *through* a voxel
contributes nothing to it. This measures the signal that counting throws away.

Every voxel of the uniform fusion is projected into every view's predicted depth
map and classified:

  supported     the view's depth along that direction lands within tol of the
                voxel's range
  contradicted  the view sees further along that direction, so it asserts the
                voxel is free space
  unobserved    the view sees something nearer (the voxel is occluded for it),
                or the direction is out of range. Never penalised.

The views that put a point in the voxel are excluded from its own score, so the
score is what *other* views say. Two questions:

  baseline    counting only views at least delta apart along the trajectory.
              Steps are 0.15 m apart, so if neighbouring views share a bias
              their agreement is not evidence, and a larger delta should
              correlate better with the true error.
  causal      offline uses every view; the ablation uses only views that come
              before the voxel's first contributor.

Reported per delta: Spearman between each score and the voxel's true error, and
the top-fraction accuracy of ranking by that score, with the recovery rate
against the oracle from step 1.

    python src/support_diag.py --mode r2 --scenes apartment_2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from data import TEST_SCENES, Sequence, sequences  # noqa: E402
from erp import quat_to_R, ray_dirs, to_radial  # noqa: E402
from eval_fusion import resize_nearest, unproject_res  # noqa: E402
from fuse import accuracy_completeness, voxel_downsample  # noqa: E402

DELTAS = (0.0, 0.3, 0.6, 1.0, 2.0)
FRACS = (0.75, 0.5, 0.25, 0.1)


def dir_to_pixel(d: np.ndarray, H: int, W: int, convention: str = "right0"):
    """Inverse of erp.ray_dirs: unit directions (N,3) in the camera frame -> (row, col) indices."""
    assert convention == "right0", "only the convention the data uses is inverted here"
    x, y, z = d[:, 0], np.clip(d[:, 1], -1, 1), d[:, 2]
    phi = np.arcsin(y)
    theta = np.arctan2(-x, -z)                      # right0: theta = -((u - 0.5) * 2 pi)
    u = 0.5 - theta / (2 * np.pi)
    v = 0.5 - phi / np.pi
    col = np.clip(np.round(u * W - 0.5).astype(np.int64), 0, W - 1)
    row = np.clip(np.round(v * H - 0.5).astype(np.int64), 0, H - 1)
    return row, col


def states(vox, origin, R, depth_radial, H, W, tol, max_depth):
    """Classify every voxel against one view: +1 supported, -1 contradicted, 0 unobserved."""
    rel = vox - origin[None, :]
    rng = np.linalg.norm(rel, axis=1)
    d_local = (rel / np.maximum(rng, 1e-9)[:, None]) @ R          # world -> camera is R^T, applied on the right
    row, col = dir_to_pixel(d_local, H, W)
    dv = depth_radial[row, col]
    out = np.zeros(len(vox), np.int8)
    ok = np.isfinite(dv) & (dv > 0) & (dv < max_depth) & (rng < max_depth)
    out[ok & (np.abs(dv - rng) <= tol)] = 1
    out[ok & (dv > rng + tol)] = -1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="r2")
    ap.add_argument("--scenes", nargs="+", default=list(TEST_SCENES))
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--tol", type=float, default=0.15, help="band that counts as agreement, metres")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--convention", default="right0")
    ap.add_argument("--causal", action="store_true", help="only views before the voxel's first contributor")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    pred_dir = a.pred_dir or REPO / "outputs" / "pred" / a.mode
    acc = {}          # (score, delta) -> list of per-sequence dicts
    rho = {}
    kept_counts = {d: [] for d in DELTAS}
    for sc in a.scenes:
        for sq in sequences(sc):
            f = pred_dir / sc / f"{sq}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            P = z["pred"].astype(np.float32); steps = z["steps"].tolist()
            S = Sequence(sc, sq)
            H, W = P.shape[1:]
            dirs = ray_dirs(H, W, a.convention)
            clouds, origins, Rs, drad = [], [], [], []
            ref = []
            for k, i in enumerate(steps):
                pts, valid, d, dw = unproject_res(P[k], S.pose(i), dirs, a.max_depth, a.stride, "face")
                clouds.append(pts)
                origins.append(np.asarray(S.pose(i)["position"], float))
                Rs.append(quat_to_R(S.pose(i)["rotation"]))
                drad.append(to_radial(P[k], dirs, "face"))
                g = resize_nearest(S.gt_depth(i, "face"), (H, W))
                ref.append(unproject_res(g, S.pose(i), dirs, a.max_depth, a.stride, "face")[0])
            ref_cloud, _ = voxel_downsample(np.concatenate(ref), a.voxel / 2)
            allP = np.concatenate(clouds)
            view_of = np.concatenate([np.full(len(c), j) for j, c in enumerate(clouds)])
            vox, _ = voxel_downsample(allP, a.voxel)
            key = np.floor(allP / a.voxel).astype(np.int64)
            _, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.ravel()
            nV, nJ = len(vox), len(clouds)
            contrib = np.zeros((nV, nJ), bool)
            contrib[inv, view_of] = True
            first = np.where(contrib.any(1), contrib.argmax(1), nJ)
            err, _ = cKDTree(ref_cloud).query(vox, k=1, workers=-1)
            St = np.stack([states(vox, origins[j], Rs[j], drad[j], H, W, a.tol, a.max_depth)
                           for j in range(nJ)], 1)                      # (nV, nJ)
            O = np.array(origins)
            for delta in DELTAS:
                # views spaced at least delta apart along the trajectory
                keep = [0]
                for j in range(1, nJ):
                    if np.linalg.norm(O[j] - O[keep[-1]]) >= delta:
                        keep.append(j)
                keep = np.array(keep)
                kept_counts[delta].append(len(keep))
                use = np.zeros((nV, nJ), bool)
                use[:, keep] = True
                use &= ~contrib                                          # other views only
                if a.causal:
                    use &= np.arange(nJ)[None, :] < first[:, None]
                sup = ((St == 1) & use).sum(1).astype(np.float64)
                con = ((St == -1) & use).sum(1).astype(np.float64)
                # the count is restricted to the same view subset, which is the point of
                # delta: if neighbouring steps 0.15 m apart share a bias, their agreement
                # is not evidence and dropping them should correlate better with the error
                cnt = (contrib[:, keep]).sum(1).astype(np.float64)
                scores = {"count": cnt,
                          "count_all_views": contrib.sum(1).astype(np.float64),
                          "sup_minus_con": sup - con,
                          "sup_ratio": np.where(sup + con > 0, sup / np.maximum(sup + con, 1), 0.0),
                          "count_plus_con": cnt - con}
                for name, sc_ in scores.items():
                    if len(np.unique(sc_)) < 2:
                        continue
                    r = spearmanr(sc_, err).correlation
                    rho.setdefault((name, delta), []).append(float(r))
                    order = np.argsort(-sc_)
                    row = {}
                    for fr in FRACS:
                        k2 = order[: max(1, int(fr * len(order)))]
                        row[str(fr)] = accuracy_completeness(vox[k2], ref_cloud, a.tau)
                    acc.setdefault((name, delta), []).append(row)
                # oracle for the recovery rate
                if delta == DELTAS[0]:
                    order = np.argsort(err)
                    row = {}
                    for fr in FRACS:
                        k2 = order[: max(1, int(fr * len(order)))]
                        row[str(fr)] = accuracy_completeness(vox[k2], ref_cloud, a.tau)
                    acc.setdefault(("oracle", 0.0), []).append(row)
            print(f"[{sc}/{sq}] {nV} voxels, {nJ} views", flush=True)

    def mean_acc(k, fr, field="acc"):
        return float(np.mean([r[str(fr)][field] for r in acc[k]]))

    out = {"mode": a.mode, "causal": a.causal, "tol": a.tol, "voxel": a.voxel, "tau": a.tau,
           "spearman": {f"{n}|{d}": float(np.mean(v)) for (n, d), v in rho.items()},
           "acc": {f"{n}|{d}": {str(fr): mean_acc((n, d), fr) for fr in FRACS} for (n, d) in acc},
           "comp": {f"{n}|{d}": {str(fr): mean_acc((n, d), fr, "comp") for fr in FRACS} for (n, d) in acc}}
    print(f"\n[{a.mode}{a.tag}] {'causal' if a.causal else 'offline'}, tol {a.tol} m, "
          f"{len(acc[('oracle', 0.0)])} sequences")
    print("Spearman(score, true error) - more negative is better (high score = low error)")
    print(f"{'score':16s}" + "".join(f"{f'delta={d}':>12s}" for d in DELTAS))
    for name in ("count", "count_all_views", "sup_minus_con", "sup_ratio", "count_plus_con"):
        cells = "".join(f"{np.mean(rho[(name, d)]):12.3f}" if (name, d) in rho else f"{'-':>12s}" for d in DELTAS)
        print(f"{name:16s}{cells}")
    print("\naccuracy of the top quarter by each score (m), and recovery against the oracle")
    orc = mean_acc(("oracle", 0.0), 0.25)
    base = mean_acc(("count_all_views", 0.0), 0.25)
    for name in ("count", "count_all_views", "sup_minus_con", "sup_ratio", "count_plus_con"):
        for d in DELTAS:
            if (name, d) not in acc:
                continue
            m = mean_acc((name, d), 0.25)
            rec = (base - m) / (base - orc) if base > orc else float("nan")
            print(f"  {name:16s} delta={d:<4} top25% {m:.3f} m  comp {mean_acc((name, d), 0.25, 'comp'):.3f}  recovery {rec:+.2f}")
    print(f"  {'oracle':16s}            top25% {orc:.3f} m  comp {mean_acc(('oracle', 0.0), 0.25, 'comp'):.3f}")
    out["n_views_kept"] = {str(d): float(np.mean(kept_counts[d])) for d in kept_counts}
    p = REPO / "outputs" / f"support_{a.mode}{'_causal' if a.causal else ''}{a.tag}.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
