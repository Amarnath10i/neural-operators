import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralops.linear import PCARidge  # noqa: E402
from neuralops.pdes import burgers, darcy, grf_1d, grf_2d_neumann  # noqa: E402
from neuralops.metrics import fourier_resample, grid_resample, rel_l2  # noqa: E402

torch = pytest.importorskip("torch")
from neuralops import numpy_fno  # noqa: E402
from neuralops.models import CNN1d, DeepONet1d, FNO1d, FNO2d, SpectralConv1d, UNet2d  # noqa: E402

X = np.arange(256) / 256


# ---------- solvers ----------

def test_burgers_reduces_to_the_heat_equation_at_small_amplitude():
    u0 = 1e-6 * np.sin(2 * np.pi * X)
    exact = u0 * np.exp(-0.1 * (2 * np.pi) ** 2)
    assert rel_l2(burgers(u0), exact[None])[0] < 1e-6


def test_burgers_conserves_the_mean_and_is_converged_in_time():
    u0 = grf_1d(3, 256, np.random.default_rng(0))
    u = burgers(u0, dt=4e-4)
    assert np.abs(u.mean(1) - u0.mean(1)).max() < 1e-12
    assert rel_l2(u, burgers(u0, dt=2e-4)).max() < 1e-8


def test_darcy_converges_at_second_order_to_a_manufactured_solution():
    errs = []
    for n in (17, 33, 65):
        g = np.linspace(0, 1, n)
        x, y = np.meshgrid(g, g, indexing="ij")
        ue = np.sin(np.pi * x) * np.sin(np.pi * y)
        a = 1 + x
        f = -(np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) - (1 + x) * 2 * np.pi ** 2 * ue)
        errs.append(np.abs(darcy(a, f) - ue).max())
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 1.8)


def test_grf_2d_has_zero_normal_derivative_at_the_walls():
    psi = grf_2d_neumann(2, 129, np.random.default_rng(0))
    # second-order one-sided derivative at the wall vs typical interior central differences
    edge = np.abs(-3 * psi[:, 0, :] + 4 * psi[:, 1, :] - psi[:, 2, :]).mean()
    interior = np.abs(psi[:, 2:, :] - psi[:, :-2, :]).mean()
    assert edge < 0.1 * interior


# ---------- resampling and the linear baseline ----------

def test_fourier_resample_is_exact_for_band_limited_signals():
    f = lambda x: np.sin(2 * np.pi * 3 * x) + 0.5 * np.cos(2 * np.pi * 7 * x)  # noqa: E731
    up = fourier_resample(f(X)[None], 1024)[0]
    assert np.allclose(up, f(np.arange(1024) / 1024), atol=1e-12)


def test_grid_resample_is_exact_for_bilinear_fields():
    g = np.linspace(0, 1, 33)
    x, y = np.meshgrid(g, g, indexing="ij")
    fine = np.linspace(0, 1, 65)
    xf, yf = np.meshgrid(fine, fine, indexing="ij")
    assert np.allclose(grid_resample((2 * x + 3 * y)[None], 65)[0], 2 * xf + 3 * yf, atol=1e-12)


def test_pca_ridge_recovers_a_linear_operator():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 20)) @ rng.standard_normal((20, 50))
    A = rng.standard_normal((50, 40))
    model = PCARidge(k_in=20, k_out=20, alpha=1e-10).fit(x[:300], x[:300] @ A)
    assert rel_l2(model.predict(x[300:]), x[300:] @ A).max() < 1e-6


# ---------- models ----------

def test_spectral_conv_output_is_band_limited():
    torch.manual_seed(0)
    conv = SpectralConv1d(3, 2, modes=8)
    out = conv(torch.randn(4, 3, 128)).detach().numpy()
    spec = np.abs(np.fft.rfft(out))
    assert spec[..., 8:].max() < 1e-4 * spec[..., :8].max()


@pytest.mark.parametrize("n", [64, 256, 1024])
def test_mesh_free_models_accept_any_resolution(n):
    u = torch.randn(2, n)
    assert FNO1d(modes=8, width=16)(u).shape == (2, n)
    assert DeepONet1d(sensors=32, p=16, width=32)(u).shape == (2, n)
    assert CNN1d(width=8, blocks=1)(u).shape == (2, n)


def test_fno2d_and_unet_shapes():
    a = torch.randn(2, 33, 33)
    assert FNO2d(modes=6, width=8)(a).shape == (2, 33, 33)
    assert UNet2d(base=8)(a).shape == (2, 33, 33)


def test_numpy_fno1d_matches_pytorch():
    torch.manual_seed(0)
    m = FNO1d(modes=8, width=16).eval()
    u = torch.randn(3, 96)
    w = numpy_fno.export(m)
    assert np.abs(numpy_fno.fno1d(w, u.numpy()) - m(u).detach().numpy()).max() < 1e-4


def test_numpy_fno2d_matches_pytorch():
    torch.manual_seed(0)
    m = FNO2d(modes=5, width=8).eval()
    a = torch.randn(2, 33, 33)
    w = numpy_fno.export(m)
    assert np.abs(numpy_fno.fno2d(w, a.numpy()) - m(a).detach().numpy()).max() < 1e-4
