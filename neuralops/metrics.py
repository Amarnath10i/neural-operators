"""Relative L2 error and resampling between grids (NumPy only, so the app can use them without PyTorch)."""
from __future__ import annotations

import numpy as np


def rel_l2(pred, true) -> np.ndarray:
    """Per-sample ||pred - true|| / ||true|| over all non-batch axes."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    axes = tuple(range(1, true.ndim))
    return np.sqrt(((pred - true) ** 2).sum(axes)) / np.sqrt((true ** 2).sum(axes))


def fourier_resample(u: np.ndarray, n: int) -> np.ndarray:
    """Band-limited (periodic) resampling of (B, N) to (B, n): exact for signals with no content above N/2."""
    N = u.shape[-1]
    if n == N:
        return u
    U = np.fft.rfft(u)
    if n < N:
        U = U[..., :n // 2 + 1]
    else:
        U = np.concatenate([U, np.zeros(U.shape[:-1] + (n // 2 + 1 - U.shape[-1],))], -1)
    return np.fft.irfft(U, n=n) * (n / N)


def grid_resample(a: np.ndarray, n: int) -> np.ndarray:
    """Bilinear resampling of (B, H, W) vertex-grid fields on [0,1]^2 to (B, n, n)."""
    from scipy.ndimage import map_coordinates

    H = a.shape[-1]
    if n == H:
        return a
    c = np.linspace(0, H - 1, n)
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return np.stack([map_coordinates(ai, [yy, xx], order=1) for ai in a])
