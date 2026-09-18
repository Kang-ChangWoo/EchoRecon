"""ERP geometry: pixel -> ray direction, radial depth -> points, habitat pose -> world.

Habitat's camera frame: +x right, +y up, -z forward. A sequence pose gives the
position (x, y, z) and a unit quaternion (w, x, y, z) rotating camera axes into
the world (rotation about +y for these sequences).

An equirectangular image of H rows and W columns maps row v to elevation
phi = (0.5 - (v + 0.5) / H) * pi (top row looks up) and column u to azimuth
theta = ((u + 0.5) / W - 0.5) * 2 pi measured from the forward axis. Which way
positive azimuth turns (left or right) and whether the centre column is the
forward or the backward axis differ between renderers, so the convention is a
parameter here and src/convention_check.py picks it from the data.
"""

from __future__ import annotations

import numpy as np

CONVENTIONS = {          # name: (azimuth sign: +1 turns left (towards -x), -1 turns right; azimuth offset)
    "left0": (+1, 0.0), "right0": (-1, 0.0), "leftpi": (+1, np.pi), "rightpi": (-1, np.pi),
}


def ray_dirs(H: int, W: int, convention: str = "right0") -> np.ndarray:
    """(H, W, 3) unit directions in the camera frame (+x right, +y up, -z forward)."""
    sign, off = CONVENTIONS[convention]
    v = (np.arange(H) + 0.5) / H
    u = (np.arange(W) + 0.5) / W
    phi = (0.5 - v) * np.pi                       # elevation, + up
    theta = sign * ((u - 0.5) * 2 * np.pi) + off  # azimuth from forward, + to the left
    ct, st = np.cos(theta)[None, :], np.sin(theta)[None, :]
    cp, sp = np.cos(phi)[:, None], np.sin(phi)[:, None]
    x = -cp * st          # left turn (+theta) moves towards -x
    y = sp * np.ones_like(ct)
    z = -cp * ct          # forward is -z
    return np.stack([x, y, z], -1)


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def unproject(depth: np.ndarray, pose: dict, convention: str = "right0", dirs: np.ndarray | None = None,
              max_depth: float = 10.0, stride: int = 1):
    """Radial ERP depth (H, W) + pose -> world points (N, 3) and the valid mask used."""
    H, W = depth.shape
    if dirs is None:
        dirs = ray_dirs(H, W, convention)
    d = depth[::stride, ::stride]
    dd = dirs[::stride, ::stride]
    valid = np.isfinite(d) & (d > 0) & (d < max_depth)
    P = dd[valid] * d[valid][:, None]
    R = quat_to_R(pose["rotation"])
    return P @ R.T + np.asarray(pose["position"])[None, :], valid
