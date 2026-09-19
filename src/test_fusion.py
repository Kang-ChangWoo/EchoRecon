#!/usr/bin/env python3
"""Unit tests that must pass before any fusion result is believed.

Conventions (a wrong one here looks exactly like a weak model downstream):
  ERP ray direction round trip, pose transform round trip, bin index against
  metric depth, voxel indexing, posterior normalisation.

And the toy problem from DIRECTION.md section 20: two views whose argmax is
wrong in different ways, but whose ground-truth modes land on the same world
voxel. Point fusion of the argmaxes must fail; posterior fusion must recover the
true surface. If this fails, the fusion implementation is wrong and no
experiment on it means anything.

    python src/test_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from erp import CONVENTIONS, quat_to_R, ray_dirs, to_radial  # noqa: E402
from posterior import ViewPosterior, candidate_grid, depth_bins, fuse, gaussian_posterior  # noqa: E402
from support_diag import dir_to_pixel  # noqa: E402

OK, FAIL = "ok", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{OK if cond else FAIL}] {name}" + (f"  {detail}" if detail else ""))


def test_erp_round_trip():
    H, W = 64, 128
    d = ray_dirs(H, W, "right0")
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    r2, c2 = dir_to_pixel(d.reshape(-1, 3), H, W)
    check("ERP direction round trip", (r2 == rows.ravel()).all() and (c2 == cols.ravel()).all(),
          f"{H*W} pixels")
    check("ERP directions are unit", np.allclose(np.linalg.norm(d, axis=-1), 1, atol=1e-6))


def test_pose_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        R = quat_to_R(q)
        if not (np.allclose(R @ R.T, np.eye(3), atol=1e-6) and abs(np.linalg.det(R) - 1) < 1e-6):
            check("pose rotation is orthonormal", False, str(q)); return
    o = rng.normal(size=3)
    p_cam = rng.normal(size=(100, 3))
    p_world = p_cam @ R.T + o
    back = (p_world - o) @ R
    check("pose rotation is orthonormal", True, "50 random quaternions")
    check("world <-> camera round trip", np.allclose(back, p_cam, atol=1e-9))


def test_depth_kind():
    H, W = 64, 128
    d = ray_dirs(H, W, "right0")
    rad = np.full((H, W), 3.0, np.float32)
    face = rad * np.abs(d).max(-1)
    check("face depth converts back to radial", np.allclose(to_radial(face, d, "face"), rad, atol=1e-6))


def test_bins():
    K = 128
    edges, centres = depth_bins(K, 0.1, 10.0)
    check("bin edges count", len(edges) == K + 1 and len(centres) == K)
    step = (edges[-1] - edges[0]) / K
    for d in (0.1, 1.0, 5.0, 9.99):
        k = int(np.clip(np.floor((d - edges[0]) / step), 0, K - 1))
        check(f"bin index brackets {d} m", edges[k] <= d <= edges[k + 1] + 1e-6,
              f"bin {k} = [{edges[k]:.3f}, {edges[k+1]:.3f}]")


def test_posterior_normalised():
    edges, _ = depth_bins(128, 0.1, 10.0)
    dep = np.array([[0.5, 3.0], [7.0, 9.5]], np.float32)
    cdf = gaussian_posterior(dep, edges, 0.2)
    check("posterior cdf starts at 0 and ends at 1",
          np.allclose(cdf[:, 0], 0, atol=1e-6) and np.allclose(cdf[:, -1], 1, atol=1e-6))
    check("posterior cdf is non-decreasing", (np.diff(cdf, axis=1) >= -1e-7).all())


def test_voxel_indexing():
    from fuse import voxel_downsample
    pts = np.array([[0.01, 0.01, 0.01], [0.09, 0.09, 0.09], [0.11, 0.0, 0.0]], np.float64)
    v, w = voxel_downsample(pts, 0.1)
    check("voxel grouping", len(v) == 2 and set(w.astype(int)) == {1, 2}, f"{len(v)} voxels, weights {w}")


def test_band_mass():
    """A delta-like posterior at 3 m must give mass 1 inside the band and 0 outside."""
    H, W = 32, 64
    edges, _ = depth_bins(256, 0.1, 10.0)
    dep = np.full((H, W), 3.0, np.float32)
    v = ViewPosterior(gaussian_posterior(dep, edges, 0.02), [0, 0, 0], [1, 0, 0, 0], H, W, edges)
    d = ray_dirs(H, W, "right0").reshape(-1, 3)
    on = d[:20] * 3.0
    off = d[:20] * 5.0
    check("band mass is 1 at the predicted range", np.all(v.band_mass(on, 0.15) > 0.99),
          f"min {v.band_mass(on, 0.15).min():.4f}")
    check("band mass is 0 far from it", np.all(v.band_mass(off, 0.15) < 1e-3))


def test_toy_two_wrong_argmaxes():
    """Two views, each with a wrong mode heavier than the true one, but whose true
    modes are the same world point. Point fusion of the argmaxes must miss it;
    posterior fusion must find it."""
    H, W = 64, 128
    K = 256
    edges, centres = depth_bins(K, 0.1, 10.0)
    xs = np.array([2.0, 0.0, -3.0])                     # the true surface point
    views = []
    argmax_pts = []
    for o in (np.array([0.0, 0.0, 0.0]), np.array([1.5, 0.0, 1.0])):
        rel = xs - o
        r_gt = np.linalg.norm(rel)
        d_local = rel / r_gt                            # identity rotation: camera == world
        row, col = dir_to_pixel(d_local[None, :], H, W)
        flat = int(row[0]) * W + int(col[0])
        # every ray gets a broad uninformative posterior; the ray towards xs gets two
        # spikes, the heavier one wrong and a different wrong range for each view
        p = np.full((H * W, K), 1.0 / K, np.float32)
        r_wrong = r_gt + (1.2 if o[0] == 0.0 else -1.1)
        k_gt = int(np.clip((r_gt - edges[0]) / (edges[-1] - edges[0]) * K, 0, K - 1))
        k_wr = int(np.clip((r_wrong - edges[0]) / (edges[-1] - edges[0]) * K, 0, K - 1))
        p[flat] = 1e-6
        p[flat, k_gt] = 0.40
        p[flat, k_wr] = 0.55
        p[flat] /= p[flat].sum()
        cdf = np.zeros((H * W, K + 1), np.float32)
        np.cumsum(p, axis=1, out=cdf[:, 1:])
        views.append(ViewPosterior(cdf, o, [1, 0, 0, 0], H, W, edges))
        argmax_pts.append(o + d_local * centres[k_wr])   # what a point pipeline would keep
    voxel = 0.1
    pts = candidate_grid(xs - 2.0, xs + 2.0, voxel)
    A = fuse(views, pts, tau=0.15)
    best = pts[int(np.argmax(A))]
    err_post = float(np.linalg.norm(best - xs))
    err_point = float(min(np.linalg.norm(np.asarray(p) - xs) for p in argmax_pts))
    check("toy: posterior fusion recovers the true surface", err_post <= 0.2,
          f"{err_post:.3f} m from the truth")
    check("toy: the argmax points do not", err_point > 0.5,
          f"nearest argmax point is {err_point:.3f} m away")


def main() -> int:
    test_erp_round_trip()
    test_pose_round_trip()
    test_depth_kind()
    test_bins()
    test_posterior_normalised()
    test_voxel_indexing()
    test_band_mass()
    test_toy_two_wrong_argmaxes()
    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    if bad:
        print("FAILED:", ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
