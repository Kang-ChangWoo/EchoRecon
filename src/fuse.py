"""Fusion of unprojected ERP depths: point clouds, voxel occupancy, weighted TSDF.

Everything is numpy; a scene of 300 steps at stride 4 (512x1024 / 16 = 32k
points a step) is ten million points, which fits.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def voxel_downsample(P: np.ndarray, size: float, W: np.ndarray | None = None):
    """Mean point per occupied voxel (weighted if W given). Returns points, weights per voxel."""
    if W is None:
        W = np.ones(len(P))
    key = np.floor(P / size).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    sw = np.bincount(inv, weights=W, minlength=n)
    out = np.stack([np.bincount(inv, weights=W * P[:, k], minlength=n) / np.maximum(sw, 1e-12) for k in range(3)], 1)
    return out, sw


# ---------------------------------------------------------------- per-ray weights
def weight_uniform(depth: np.ndarray, **_):
    return np.ones_like(depth, dtype=np.float32)


def weight_range_prior(depth: np.ndarray, scale: float = 3.0, power: float = 2.0, **_):
    """The trivial baseline: reliability as a fixed decreasing function of range only,
    w = 1 / (1 + (d / scale) ** power). Any confidence that does not beat this has
    reduced to a distance prior."""
    return (1.0 / (1.0 + (depth / scale) ** power)).astype(np.float32)


WEIGHTS = {"uniform": weight_uniform, "range_prior": weight_range_prior}


def accuracy_completeness(pred: np.ndarray, gt: np.ndarray, tau: float = 0.1):
    """Chamfer-style: accuracy = mean dist pred->gt and fraction within tau; completeness = gt->pred."""
    if len(pred) == 0 or len(gt) == 0:
        return dict(acc=np.nan, acc_frac=np.nan, comp=np.nan, comp_frac=np.nan)
    d_pg, _ = cKDTree(gt).query(pred, k=1, workers=-1)
    d_gp, _ = cKDTree(pred).query(gt, k=1, workers=-1)
    return dict(acc=float(d_pg.mean()), acc_frac=float((d_pg < tau).mean()),
                comp=float(d_gp.mean()), comp_frac=float((d_gp < tau).mean()),
                chamfer=float(0.5 * (d_pg.mean() + d_gp.mean())))


class TSDF:
    """Weighted truncated signed distance volume over an axis-aligned box."""

    def __init__(self, lo, hi, voxel: float = 0.1, trunc: float = 0.3):
        self.lo = np.asarray(lo, float); self.voxel = voxel; self.trunc = trunc
        self.shape = np.ceil((np.asarray(hi) - self.lo) / voxel).astype(int) + 1
        self.D = np.zeros(self.shape, np.float32)
        self.W = np.zeros(self.shape, np.float32)

    def integrate(self, origin: np.ndarray, dirs: np.ndarray, depth: np.ndarray, weight: np.ndarray | None = None):
        """Rays from `origin` along unit `dirs` (N,3) hitting at `depth` (N,), per-ray weights."""
        if weight is None:
            weight = np.ones(len(depth), np.float32)
        step = self.voxel * 0.5
        # samples along each ray from depth - trunc to depth + trunc
        offs = np.arange(-self.trunc, self.trunc + step, step, dtype=np.float32)
        t = depth[:, None] + offs[None, :]                              # (N, S)
        ok = t > 0
        P = origin[None, None, :] + dirs[:, None, :] * t[:, :, None]      # (N, S, 3)
        sdf = np.clip((depth[:, None] - t) / self.trunc, -1, 1)            # + in front of the surface
        idx = np.round((P - self.lo) / self.voxel).astype(np.int64)
        inb = ok & np.all((idx >= 0) & (idx < self.shape), -1)
        idx = idx[inb]; sdf = sdf[inb]; w = np.broadcast_to(weight[:, None], t.shape)[inb]
        flat = np.ravel_multi_index(idx.T, self.shape)
        wsum = np.bincount(flat, weights=w, minlength=self.D.size)
        dsum = np.bincount(flat, weights=w * sdf, minlength=self.D.size)
        D, W = self.D.ravel(), self.W.ravel()
        D[:] = (D * W + dsum) / np.maximum(W + wsum, 1e-12)
        W[:] = W + wsum

    def surface_points(self, min_w: float = 1.0):
        """Voxel centres where the signed distance crosses zero (a cheap surface extraction)."""
        D, W = self.D, self.W
        near = (np.abs(D) < 0.5) & (W >= min_w)
        idx = np.argwhere(near)
        return idx * self.voxel + self.lo, W[near]
