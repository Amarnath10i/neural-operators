"""FNO inference in NumPy, so trained models run without PyTorch (and in the browser via Pyodide).

`export(model)` (needs PyTorch) turns a trained FNO1d/FNO2d into a dict of
arrays saved with np.savez; `fno1d` / `fno2d` evaluate it. tests/ check that
both give the same output as the PyTorch model.
"""
from __future__ import annotations

import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def export(model) -> dict[str, np.ndarray]:
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    sd["kind"] = np.array("fno2d" if "spec.0.w_pos" in sd else "fno1d")
    if hasattr(model, "pad_frac"):
        sd["pad_frac"] = np.array(model.pad_frac)
    return sd


def _layers(w):
    return sorted({int(k.split(".")[1]) for k in w if k.startswith("spec.")})


def fno1d(w: dict, u: np.ndarray) -> np.ndarray:
    """u: (B, N) -> (B, N)."""
    u = np.atleast_2d(u).astype(np.float64)
    b, n = u.shape
    x = np.broadcast_to(np.linspace(0, 1, n + 1)[:-1], u.shape)
    h = np.stack([u, x], -1) @ w["lift.weight"].T + w["lift.bias"]  # (B, N, width)
    h = h.transpose(0, 2, 1)  # (B, width, N)
    layers = _layers(w)
    for i in layers:
        W = w[f"spec.{i}.weight"]  # (in, out, modes) complex
        h_ft = np.fft.rfft(h)
        m = min(W.shape[-1], h_ft.shape[-1])
        out = np.zeros((b, W.shape[1], h_ft.shape[-1]), dtype=complex)
        out[..., :m] = np.einsum("bix,iox->box", h_ft[..., :m], W[..., :m])
        spec = np.fft.irfft(out, n=n)
        pw = np.einsum("oi,bin->bon", w[f"pw.{i}.weight"][..., 0], h) + w[f"pw.{i}.bias"][None, :, None]
        h = spec + pw
        if i != layers[-1]:
            h = gelu(h)
    h = gelu(h.transpose(0, 2, 1) @ w["proj1.weight"].T + w["proj1.bias"])
    return (h @ w["proj2.weight"].T + w["proj2.bias"])[..., 0]


def fno2d(w: dict, a: np.ndarray) -> np.ndarray:
    """a: (B, H, W) -> (B, H, W)."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 2:
        a = a[None]
    b, hgt, wid = a.shape
    gy, gx = np.meshgrid(np.linspace(0, 1, hgt), np.linspace(0, 1, wid), indexing="ij")
    inp = np.stack([a, np.broadcast_to(gy, a.shape), np.broadcast_to(gx, a.shape)], -1)
    h = (inp @ w["lift.weight"].T + w["lift.bias"]).transpose(0, 3, 1, 2)
    p = int(round(float(w["pad_frac"]) * (hgt - 1)))
    h = np.pad(h, ((0, 0), (0, 0), (0, p), (0, p)))
    H, W_ = h.shape[-2:]
    layers = _layers(w)
    for i in layers:
        wp, wn = w[f"spec.{i}.w_pos"], w[f"spec.{i}.w_neg"]
        h_ft = np.fft.rfft2(h)
        out = np.zeros((b, wp.shape[1], H, W_ // 2 + 1), dtype=complex)
        m1, m2 = min(wp.shape[2], H // 2), min(wp.shape[3], W_ // 2 + 1)
        out[:, :, :m1, :m2] = np.einsum("bixy,ioxy->boxy", h_ft[:, :, :m1, :m2], wp[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = np.einsum("bixy,ioxy->boxy", h_ft[:, :, -m1:, :m2], wn[:, :, :m1, :m2])
        spec = np.fft.irfft2(out, s=(H, W_))
        pw = np.einsum("oi,bihw->bohw", w[f"pw.{i}.weight"][..., 0, 0], h) + w[f"pw.{i}.bias"][None, :, None, None]
        h = spec + pw
        if i != layers[-1]:
            h = gelu(h)
    h = h[..., :hgt, :wid].transpose(0, 2, 3, 1)
    h = gelu(h @ w["proj1.weight"].T + w["proj1.bias"])
    return (h @ w["proj2.weight"].T + w["proj2.bias"])[..., 0]


def load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def predict(w: dict, x: np.ndarray) -> np.ndarray:
    return fno2d(w, x) if str(w["kind"]) == "fno2d" else fno1d(w, x)
