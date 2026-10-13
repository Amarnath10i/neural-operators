"""Data for the two benchmark operators, generated with numerical solvers that are tested in tests/.

Burgers (Li et al., 2021 setup): u_t + (u^2/2)_x = nu u_xx on the periodic
interval [0, 1), nu = 0.1; the operator maps u(., 0) to u(., 1). Initial
conditions are Gaussian random fields. Solved pseudo-spectrally with an
integrating factor for the viscous term, RK4 in time and 2/3 dealiasing.

Darcy: -div(a grad u) = 1 on the unit square with u = 0 on the boundary; the
operator maps the permeability a to the pressure u. a is 12 where a Gaussian
random field is positive and 3 elsewhere. Solved with the standard 5-point
finite-volume scheme on a vertex grid (harmonic means of a on cell faces),
so grids of 129, 65 and 33 points per side are nested and subsampling the
fine solution gives exact coarse-grid inputs and targets.

Everything is NumPy (+ SciPy for the sparse solve), so the solvers also run
in the browser for the demo app.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- Burgers

def grf_1d(n_samples: int, n: int, rng: np.random.Generator, alpha: float = 2.5, tau: float = 7.0,
           sigma: float = 160.0) -> np.ndarray:
    """Periodic Gaussian random fields on n points: covariance sigma^2 (-Laplacian + tau^2)^(-alpha)."""
    k = np.arange(n // 2 + 1)
    coef = sigma * (4 * np.pi ** 2 * k ** 2 + tau ** 2) ** (-alpha / 2)
    coef[0] = 0.0  # zero mean
    z = rng.standard_normal((n_samples, k.size)) + 1j * rng.standard_normal((n_samples, k.size))
    return np.fft.irfft(z * coef * n / np.sqrt(2), n=n)


def burgers(u0: np.ndarray, nu: float = 0.1, t_end: float = 1.0, dt: float = 2e-4) -> np.ndarray:
    """u(., t_end) for periodic viscous Burgers on [0, 1), batched over the leading axis."""
    u0 = np.atleast_2d(u0)
    n = u0.shape[-1]
    k = 2 * np.pi * np.fft.rfftfreq(n, d=1.0 / n)
    dealias = np.abs(np.fft.rfftfreq(n, d=1.0 / n)) < n / 3
    lin = -nu * k ** 2

    def nonlinear(v_hat):
        u = np.fft.irfft(v_hat, n=n)
        return -0.5j * k * np.fft.rfft(u * u) * dealias

    steps = int(round(t_end / dt))
    dt = t_end / steps
    e_half, e_full = np.exp(lin * dt / 2), np.exp(lin * dt)
    v = np.fft.rfft(u0)
    for _ in range(steps):
        a = nonlinear(v)
        b = nonlinear(e_half * (v + dt / 2 * a))
        c = nonlinear(e_half * v + dt / 2 * b)
        d = nonlinear(e_full * v + dt * e_half * c)
        v = e_full * v + dt / 6 * (e_full * a + 2 * e_half * (b + c) + d)
    return np.fft.irfft(v, n=n)


def make_burgers(n_samples: int, n: int = 1024, seed: int = 0, batch: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u0 = grf_1d(n_samples, n, rng)
    u1 = np.concatenate([burgers(u0[i:i + batch]) for i in range(0, n_samples, batch)])
    return u0.astype(np.float32), u1.astype(np.float32)


# ---------------------------------------------------------------- Darcy

def grf_2d_neumann(n_samples: int, n: int, rng: np.random.Generator, alpha: float = 2.0, tau: float = 3.0,
                   modes: int = 32) -> np.ndarray:
    """Gaussian random fields on an n x n vertex grid of [0,1]^2 with a cosine (zero-Neumann) basis."""
    x = np.linspace(0, 1, n)
    kk = np.arange(modes)
    basis = np.cos(np.pi * np.outer(x, kk))  # (n, modes)
    lam = np.pi ** 2 * (kk[:, None] ** 2 + kk[None, :] ** 2) + tau ** 2
    coef = tau ** (alpha - 1) * lam ** (-alpha / 2)
    coef[0, 0] = 0.0
    xi = rng.standard_normal((n_samples, modes, modes))
    return np.einsum("im,smn,jn->sij", basis, xi * coef, basis)


def darcy_coefficient(psi: np.ndarray, high: float = 12.0, low: float = 3.0) -> np.ndarray:
    return np.where(psi >= 0, high, low)


def darcy(a: np.ndarray, f: float | np.ndarray = 1.0):
    """Solve -div(a grad u) = f with u = 0 on the boundary; a and f on an n x n vertex grid."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    n = a.shape[0]
    h = 1.0 / (n - 1)
    m = n - 2  # interior points per side
    f_int = (np.broadcast_to(f, a.shape))[1:-1, 1:-1]
    idx = np.arange(m * m).reshape(m, m)

    def face(a1, a2):
        return 2 * a1 * a2 / (a1 + a2)

    ac = a[1:-1, 1:-1]
    a_e, a_w = face(ac, a[1:-1, 2:]), face(ac, a[1:-1, :-2])
    a_n, a_s = face(ac, a[2:, 1:-1]), face(ac, a[:-2, 1:-1])
    rows, cols, vals = [idx.ravel()], [idx.ravel()], [(a_e + a_w + a_n + a_s).ravel()]
    for coef, di, dj in ((a_e, 0, 1), (a_w, 0, -1), (a_n, 1, 0), (a_s, -1, 0)):
        ii, jj = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        ni, nj = ii + di, jj + dj
        ok = (ni >= 0) & (ni < m) & (nj >= 0) & (nj < m)  # neighbours on the boundary are 0
        rows.append(idx[ii[ok], jj[ok]])
        cols.append(idx[ni[ok], nj[ok]])
        vals.append(-coef[ok])
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m * m, m * m)).tocsr()
    u = np.zeros_like(a, dtype=float)
    u[1:-1, 1:-1] = spsolve(A, h * h * f_int.ravel()).reshape(m, m)
    return u


def make_darcy(n_samples: int, n: int = 129, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = darcy_coefficient(grf_2d_neumann(n_samples, n, rng))
    u = np.stack([darcy(ai) for ai in a])
    return a.astype(np.float32), u.astype(np.float32)
