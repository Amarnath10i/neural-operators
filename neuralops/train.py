"""Training loop and batched prediction (PyTorch)."""
from __future__ import annotations

import time

import numpy as np
import torch

from .metrics import fourier_resample, grid_resample, rel_l2  # noqa: F401  (re-exported)


def fit(model, x, y, epochs=500, batch=20, lr=1e-3, wd=1e-4, step=100, gamma=0.5, device="cuda", log=print, seed=0):
    """Adam on the relative L2 loss with step decay (the schedule of Li et al., 2021). x, y: float32 arrays."""
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step, gamma=gamma)
    xt, yt = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        tot = 0.0
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch].to(device)
            pred = model(xt[idx])
            diff = (pred - yt[idx]).flatten(1).norm(dim=1)
            loss = (diff / yt[idx].flatten(1).norm(dim=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if (ep + 1) % max(1, epochs // 5) == 0:
            log(f"    epoch {ep + 1}/{epochs}  train rel L2 {tot / len(xt):.4f}  {time.time() - t0:.0f}s")
    return model


@torch.no_grad()
def predict(model, x, batch=50, device="cuda") -> np.ndarray:
    model.eval()
    out = [model(torch.as_tensor(x[i:i + batch], device=device)).cpu().numpy() for i in range(0, len(x), batch)]
    return np.concatenate(out)
