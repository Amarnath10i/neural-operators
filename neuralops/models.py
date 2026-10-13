"""Operator-learning models compared in the benchmark (PyTorch).

- FNO (Li et al., ICLR 2021): lifting, Fourier layers that multiply the lowest
  modes by learned complex weights plus a pointwise linear path, projection.
  Weights live in frequency space, so the same model accepts any grid size.
- DeepONet (Lu et al., Nature MI 2021): a branch net reads the input at fixed
  sensor locations, a trunk net reads the query coordinate; the output is
  their dot product, so it can be queried anywhere.
- CNN / U-Net: convolutional baselines whose kernels are tied to the grid
  spacing they were trained on.

GELU uses the tanh approximation so the NumPy reimplementation in
neuralops/numpy_fno.py can match it exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    return F.gelu(x, approximate="tanh")


class SpectralConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1 / (c_in * c_out)
        self.weight = nn.Parameter(scale * torch.randn(c_in, c_out, modes, dtype=torch.cfloat))

    def forward(self, x):  # (B, C, N)
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x)
        m = min(self.modes, x_ft.shape[-1])
        out = torch.zeros(x.shape[0], self.weight.shape[1], x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out[..., :m] = torch.einsum("bix,iox->box", x_ft[..., :m], self.weight[..., :m])
        return torch.fft.irfft(out, n=n)


class SpectralConv2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, modes1: int, modes2: int):
        super().__init__()
        self.m1, self.m2 = modes1, modes2
        scale = 1 / (c_in * c_out)
        self.w_pos = nn.Parameter(scale * torch.randn(c_in, c_out, modes1, modes2, dtype=torch.cfloat))
        self.w_neg = nn.Parameter(scale * torch.randn(c_in, c_out, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):  # (B, C, H, W)
        h, w = x.shape[-2:]
        x_ft = torch.fft.rfft2(x)
        out = torch.zeros(x.shape[0], self.w_pos.shape[1], h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1, m2 = min(self.m1, h // 2), min(self.m2, w // 2 + 1)
        out[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.w_pos[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.w_neg[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(h, w))


class FNO1d(nn.Module):
    """Input (B, N) on a uniform periodic grid; output (B, N)."""

    def __init__(self, modes: int = 16, width: int = 64, layers: int = 4):
        super().__init__()
        self.lift = nn.Linear(2, width)
        self.spec = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.proj1, self.proj2 = nn.Linear(width, 128), nn.Linear(128, 1)

    def forward(self, u):
        n = u.shape[-1]
        x = torch.linspace(0, 1, n + 1, device=u.device)[:-1].expand_as(u)
        h = self.lift(torch.stack([u, x], -1)).permute(0, 2, 1)
        for i, (s, w) in enumerate(zip(self.spec, self.pw)):
            h = s(h) + w(h)
            if i < len(self.spec) - 1:
                h = gelu(h)
        return self.proj2(gelu(self.proj1(h.permute(0, 2, 1)))).squeeze(-1)


class FNO2d(nn.Module):
    """Input (B, H, W) on a vertex grid of [0,1]^2; output (B, H, W).

    Darcy is not periodic, so the domain is zero-padded by `pad_frac` of its
    size before the Fourier layers; padding a fraction (not a fixed number of
    points) keeps the model consistent across resolutions.
    """

    def __init__(self, modes: int = 12, width: int = 32, layers: int = 4, pad_frac: float = 0.125):
        super().__init__()
        self.pad_frac = pad_frac
        self.lift = nn.Linear(3, width)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj1, self.proj2 = nn.Linear(width, 128), nn.Linear(128, 1)

    def forward(self, a):
        b, hgt, wid = a.shape
        gy, gx = torch.meshgrid(torch.linspace(0, 1, hgt, device=a.device),
                                torch.linspace(0, 1, wid, device=a.device), indexing="ij")
        inp = torch.stack([a, gy.expand_as(a), gx.expand_as(a)], -1)
        h = self.lift(inp).permute(0, 3, 1, 2)
        p = int(round(self.pad_frac * (hgt - 1)))
        h = F.pad(h, (0, p, 0, p))
        for i, (s, w) in enumerate(zip(self.spec, self.pw)):
            h = s(h) + w(h)
            if i < len(self.spec) - 1:
                h = gelu(h)
        h = h[..., :hgt, :wid]
        return self.proj2(gelu(self.proj1(h.permute(0, 2, 3, 1)))).squeeze(-1)


class DeepONet1d(nn.Module):
    """Branch on the input linearly interpolated to `sensors` points; trunk on the query coordinate."""

    def __init__(self, sensors: int = 128, p: int = 128, width: int = 256):
        super().__init__()
        self.sensors = sensors
        self.branch = nn.Sequential(nn.Linear(sensors, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                    nn.Linear(width, p))
        # periodic trunk: features of x through sin/cos of the first harmonics
        self.trunk = nn.Sequential(nn.Linear(16, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                   nn.Linear(width, p))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u):
        n = u.shape[-1]
        src = torch.cat([u, u[:, :1]], -1)  # periodic wrap for interpolation
        pos = torch.linspace(0, n, self.sensors + 1, device=u.device)[:-1]
        i0 = pos.floor().long()
        t = pos - i0
        sensed = src[:, i0] * (1 - t) + src[:, i0 + 1] * t
        x = torch.linspace(0, 1, n + 1, device=u.device)[:-1]
        k = torch.arange(1, 9, device=u.device) * 2 * torch.pi
        feats = torch.cat([torch.sin(x[:, None] * k), torch.cos(x[:, None] * k)], -1)
        return self.branch(sensed) @ self.trunk(feats).T + self.bias


class CNN1d(nn.Module):
    """Residual 1D CNN with circular padding (periodic domain)."""

    def __init__(self, width: int = 64, blocks: int = 8, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.inp = nn.Conv1d(1, width, kernel, padding=pad, padding_mode="circular")
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv1d(width, width, kernel, padding=pad, padding_mode="circular"), nn.GELU(),
            nn.Conv1d(width, width, kernel, padding=pad, padding_mode="circular")) for _ in range(blocks)])
        self.out = nn.Conv1d(width, 1, 1)

    def forward(self, u):
        h = self.inp(u[:, None])
        for b in self.blocks:
            h = h + b(gelu(h))
        return self.out(gelu(h))[:, 0]


class UNet2d(nn.Module):
    """Small U-Net (3 levels) for the Darcy baseline; inputs padded to a multiple of 4."""

    def __init__(self, base: int = 32):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GELU(), nn.Conv2d(co, co, 3, padding=1), nn.GELU())
        self.e1, self.e2, self.e3 = block(1, base), block(base, 2 * base), block(2 * base, 4 * base)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, 2), block(4 * base, 2 * base)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * base, base, 2, 2), block(2 * base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, a):
        h, w = a.shape[-2:]
        H, W = -(-h // 4) * 4, -(-w // 4) * 4
        x = F.pad(a[:, None], (0, W - w, 0, H - h))
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)[:, 0, :h, :w]


def n_params(model: nn.Module) -> int:
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
