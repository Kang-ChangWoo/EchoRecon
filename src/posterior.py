"""Ray-wise depth posteriors and the soft surface-consensus fusion that uses them.

A posterior for one view is a categorical distribution over K depth bins for
every ERP ray, stored as its cumulative sum so that the probability mass inside
a band can be read with two lookups.

Fusion (surface consensus, no free-space carving; see DIRECTION.md section 10):
for a world voxel x and a view i with origin o_i,

    r_i(x) = |x - o_i|                       range from that view
    u_i(x) = the ERP ray through x           direction, in the view's frame
    O_i(x) = sum of p_i(d | u_i(x)) over |d - r_i(x)| <= tau
    A(x)   = sum_i w_i O_i(x)

Bins are linear in metric depth: an active echo's time of flight is linear in
range, so there is no reason to space them in inverse depth.
"""

from __future__ import annotations

import numpy as np

from erp import face_cos, quat_to_R, ray_dirs, to_radial
from support_diag import dir_to_pixel


def depth_bins(K: int = 128, d_min: float = 0.1, d_max: float = 10.0):
    """(K+1 edges, K centres), linear in metric depth."""
    edges = np.linspace(d_min, d_max, K + 1, dtype=np.float32)
    return edges, 0.5 * (edges[:-1] + edges[1:])


def gaussian_posterior(depth_radial: np.ndarray, edges: np.ndarray, sigma: float) -> np.ndarray:
    """Fake posterior: a Gaussian of fixed width around the point prediction.

    Returns the cumulative distribution over bins, shape (H*W, K+1), so that
    band mass is cdf[:, hi] - cdf[:, lo]. Normalised over the bin range, which
    is what a categorical head would also produce.
    """
    from scipy.special import erf
    d = depth_radial.reshape(-1, 1).astype(np.float32)
    z = (edges[None, :] - d) / (np.sqrt(2.0) * sigma)
    cdf = 0.5 * (1.0 + erf(z))
    cdf -= cdf[:, :1]
    tot = np.maximum(cdf[:, -1:], 1e-12)
    return (cdf / tot).astype(np.float32)


def logits_to_cdf(logits: np.ndarray) -> np.ndarray:
    """(H*W, K) logits -> cumulative distribution (H*W, K+1)."""
    x = logits - logits.max(-1, keepdims=True)
    p = np.exp(x); p /= np.maximum(p.sum(-1, keepdims=True), 1e-12)
    out = np.zeros((len(p), p.shape[1] + 1), np.float32)
    np.cumsum(p, axis=-1, out=out[:, 1:])
    return out


class ViewPosterior:
    """One view's posterior plus the pose needed to query it from world space.

    `space` says what the bins measure. "radial" is Euclidean distance along the
    ray. "face" is the per-face cubemap z-depth the base model was trained on,
    which is radial * max(|dx|,|dy|,|dz|); the factor depends only on the ray, so
    a radial query band [r-tau, r+tau] becomes [(r-tau)c, (r+tau)c] in face
    space. Converting the query rather than the posterior keeps it exact. Getting
    this wrong is not subtle: the factor reaches sqrt(3) at a cube corner, which
    at 5 m is 3.7 m, about 24 times the usual band.
    """

    def __init__(self, cdf: np.ndarray, origin, rotation, H: int, W: int, edges: np.ndarray,
                 space: str = "radial", rounding: str = "interp"):
        assert space in ("radial", "face") and rounding in ("interp", "whole")
        self.cdf = cdf; self.H = H; self.W = W; self.edges = edges; self.space = space
        self.rounding = rounding    # "whole": count only bins wholly inside the band (the old behaviour)
        self.origin = np.asarray(origin, float)
        self.R = quat_to_R(rotation) if np.asarray(rotation).size == 4 else np.asarray(rotation)
        self.fc = face_cos(ray_dirs(H, W)).reshape(-1) if space == "face" else None

    def band_mass(self, pts: np.ndarray, tau: float, chunk: int = 200_000) -> np.ndarray:
        """Probability that this view places a surface within tau of each point's range."""
        out = np.zeros(len(pts), np.float32)
        K = self.cdf.shape[1] - 1
        lo_e, hi_e = self.edges[0], self.edges[-1]
        step = (hi_e - lo_e) / K
        for s in range(0, len(pts), chunk):
            p = pts[s:s + chunk]
            rel = p - self.origin[None, :]
            rng = np.linalg.norm(rel, axis=1)
            d_local = (rel / np.maximum(rng, 1e-9)[:, None]) @ self.R
            row, col = dir_to_pixel(d_local, self.H, self.W)
            flat = row * self.W + col
            r_lo, r_hi = rng - tau, rng + tau
            if self.space == "face":
                c = self.fc[flat]
                r_lo, r_hi = r_lo * c, r_hi * c
            # the cdf is known at bin edges; inside a bin the mass is taken as
            # uniform, so the band's ends are interpolated rather than rounded to
            # whole bins. Rounding counted only bins wholly inside the band, which
            # at K=128 and tau=0.15 shrank the band to about 2.3 of its nominal
            # 3.9 bins in face space and to one bin or less on 6 % of rays.
            u_lo = np.clip((r_lo - lo_e) / step, 0.0, float(K))
            u_hi = np.clip((r_hi - lo_e) / step, 0.0, float(K))
            if self.rounding == "whole":
                lo = np.ceil(u_lo).astype(np.int64); hi = np.floor(u_hi).astype(np.int64)
                m = self.cdf[flat, hi] - self.cdf[flat, lo]
                out[s:s + chunk] = np.where(hi > lo, m, 0.0)
            else:
                out[s:s + chunk] = self._cdf_at(flat, u_hi) - self._cdf_at(flat, u_lo)
        return out

    def _cdf_at(self, flat: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Cumulative mass at a continuous bin coordinate u in [0, K]."""
        K = self.cdf.shape[1] - 1
        k = np.minimum(np.floor(u).astype(np.int64), K - 1)
        f = (u - k).astype(np.float32)
        c0 = self.cdf[flat, k]
        return c0 + f * (self.cdf[flat, k + 1] - c0)


def fuse(views, pts: np.ndarray, tau: float = 0.15, weights=None) -> np.ndarray:
    """Surface evidence A(x) at each point, summed over views."""
    w = np.ones(len(views)) if weights is None else np.asarray(weights, float)
    A = np.zeros(len(pts), np.float32)
    for wi, v in zip(w, views):
        A += wi * v.band_mass(pts, tau)
    return A


def candidate_grid(bounds_lo, bounds_hi, voxel: float) -> np.ndarray:
    """Voxel centres filling a box, the candidate set posterior fusion scores."""
    axes = [np.arange(lo, hi + voxel, voxel, dtype=np.float32) for lo, hi in zip(bounds_lo, bounds_hi)]
    g = np.meshgrid(*axes, indexing="ij")
    return np.stack([a.ravel() for a in g], 1)


def view_from_prediction(depth_face: np.ndarray, pose: dict, edges: np.ndarray, sigma: float,
                         convention: str = "right0") -> ViewPosterior:
    """A fake-Gaussian view posterior built from a point depth prediction."""
    H, W = depth_face.shape
    dirs = ray_dirs(H, W, convention)
    rad = to_radial(depth_face, dirs, "face")
    return ViewPosterior(gaussian_posterior(rad, edges, sigma), pose["position"], pose["rotation"], H, W, edges)
