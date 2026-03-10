"""
src/selectors.py
================
Observation grid selectors for the WS experiments.

All selectors return (ox, oy, oz): three integer arrays of 0-based
grid indices with the same length (nobs,).
"""

import numpy as np
from typing import Tuple

IntArrayTriple = Tuple[np.ndarray, np.ndarray, np.ndarray]


def uniform_3d(nx: int, ny: int, nz: int,
               stride: int = 2) -> IntArrayTriple:
    """
    Uniform grid on the full 3-D domain, one point every `stride` cells
    in each dimension.

    This is the selector for WS-3 and WS-4.
    """
    xs = np.arange(0, nx, max(1, stride))
    ys = np.arange(0, ny, max(1, stride))
    zs = np.arange(0, nz, max(1, stride))
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return (gx.ravel().astype(int),
            gy.ravel().astype(int),
            gz.ravel().astype(int))


def single_point(x: int, y: int, z: int) -> IntArrayTriple:
    """Single observation at (x, y, z). Used for WS-1 and WS-2."""
    return (np.array([x], dtype=int),
            np.array([y], dtype=int),
            np.array([z], dtype=int))


# ── Legacy 2-D selectors (kept for backward compatibility) ────────────────

def full2d(nx: int, nz: int) -> IntArrayTriple:
    """All points on a 2-D x-z cross-section (ny=1)."""
    ox, oz = np.where(np.ones((nx, nz), dtype=bool))
    return ox.astype(int), np.zeros_like(ox), oz.astype(int)


def every_other(nx: int, nz: int,
                stride_x: int = 2, stride_z: int = 2) -> IntArrayTriple:
    """Strided 2-D selector on a x-z cross-section (ny=1)."""
    xs = np.arange(0, nx, max(1, stride_x))
    zs = np.arange(0, nz, max(1, stride_z))
    mask = np.zeros((nx, nz), dtype=bool)
    mask[np.ix_(xs, zs)] = True
    ox, oz = np.where(mask)
    return ox.astype(int), np.zeros_like(ox), oz.astype(int)
