"""
Evaluation Metrics for Neural Operators.

Implements standard metrics for PDE solution accuracy:
- Relative L2 error
- Spectral error
- Energy error
- Zero-shot super-resolution evaluation
- Domain adaptation benchmarks
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import math


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Relative L2 error: ||pred - target||_2 / ||target||_2
    
    Args:
        pred: Predicted tensor
        target: Ground truth tensor
        dim: Dimension(s) to compute norm over (default: all spatial dims)
    
    Returns:
        Relative L2 error per sample
    """
    diff = pred - target
    
    if dim == -1:
        # All spatial dimensions
        spatial_dims = tuple(range(1, diff.ndim))
        diff_norm = torch.norm(diff.flatten(1), dim=1)
        target_norm = torch.norm(target.flatten(1), dim=1)
    else:
        diff_norm = torch.norm(diff, dim=dim)
        target_norm = torch.norm(target, dim=dim)
    
    return diff_norm / (target_norm + 1e-8)


def relative_l1_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L1 error."""
    diff = pred - target
    return torch.norm(diff.flatten(1), p=1, dim=1) / (torch.norm(target.flatten(1), p=1, dim=1) + 1e-8)


def spectral_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Spectral error in Fourier space.
    Measures error per wavenumber mode.
    """
    pred_ft = torch.fft.rfftn(pred, dim=tuple(range(1, pred.ndim)))
    target_ft = torch.fft.rfftn(target, dim=tuple(range(1, target.ndim)))
    
    diff_ft = pred_ft - target_ft
    
    # Error per mode
    error_per_mode = torch.abs(diff_ft) / (torch.abs(target_ft) + 1e-8)
    
    # Average over batch
    return error_per_mode.mean(0)


def spectral_convergence_rate(
    pred: torch.Tensor, 
    target: torch.Tensor,
    k_max: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spectral convergence rate.
    Returns (wavenumbers, error_spectrum).
    """
    pred_ft = torch.fft.rfftn(pred, dim=tuple(range(1, pred.ndim)))
    target_ft = torch.fft.rfftn(target, dim=tuple(range(1, target.ndim)))
    
    # Radial averaging of spectrum
    n = pred.shape[-1]
    k = torch.fft.fftfreq(n) * n
    k = k[:n//2+1]
    
    if pred.ndim == 3:  # 2D
        ky = torch.fft.rfftfreq(n) * n
        KX, KY = torch.meshgrid(k, ky, indexing='ij')
        K = torch.sqrt(KX**2 + KY**2)
    elif pred.ndim == 4:  # 3D
        ky = torch.fft.rfftfreq(n) * n
        kz = torch.fft.rfftfreq(n) * n
        KX, KY, KZ = torch.meshgrid(k, ky, kz, indexing='ij')
        K = torch.sqrt(KX**2 + KY**2 + KZ**2)
    else:  # 1D
        K = k
    
    diff_ft = pred_ft - target_ft
    error_spectrum = torch.abs(diff_ft)**2
    
    # Radial binning
    if pred.ndim > 2:
        K_flat = K.flatten()
        error_flat = error_spectrum.flatten(1).mean(0)  # Average over batch
        
        k_max = k_max or int(n//2)
        bins = torch.arange(k_max + 1, device=pred.device)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        error_binned = torch.zeros_like(bin_centers, dtype=torch.float)
        counts = torch.zeros_like(bin_centers)
        
        for i in range(len(bin_centers)):
            mask = (K_flat >= bins[i]) & (K_flat < bins[i+1])
            if mask.any():
                error_binned[i] = error_flat[mask].mean()
                counts[i] = mask.sum()
        
        return bin_centers[counts > 0], error_binned[counts > 0]
    else:
        return k, error_spectrum.mean(0)


def energy_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Energy error for PDEs: relative error in conserved quantities.
    For Navier-Stokes: kinetic energy, enstrophy.
    """
    # Kinetic energy
    pred_energy = 0.5 * torch.sum(pred**2, dim=tuple(range(1, pred.ndim)))
    target_energy = 0.5 * torch.sum(target**2, dim=tuple(range(1, target.ndim)))
    
    return torch.abs(pred_energy - target_energy) / (target_energy + 1e-8)


def enstrophy_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Enstrophy error (for Navier-Stokes vorticity)."""
    # Enstrophy = 0.5 * ||∇w||^2
    # In Fourier space: sum k^2 |w_k|^2
    pred_ft = torch.fft.rfftn(pred, dim=tuple(range(1, pred.ndim)))
    target_ft = torch.fft.rfftn(target, dim=tuple(range(1, target.ndim)))
    
    n = pred.shape[-1]
    k = torch.fft.fftfreq(n) * n
    
    if pred.ndim == 3:  # 2D
        ky = torch.fft.rfftfreq(n) * n
        KX, KY = torch.meshgrid(k, ky, indexing='ij')
        K2 = KX**2 + KY**2
    elif pred.ndim == 4:  # 3D
        ky = torch.fft.rfftfreq(n) * n
        kz = torch.fft.rfftfreq(n) * n
        KX, KY, KZ = torch.meshgrid(k, ky, kz, indexing='ij')
        K2 = KX**2 + KY**2 + KZ**2
    else:
        K2 = k**2
    
    pred_enstrophy = 0.5 * torch.sum(K2 * torch.abs(pred_ft)**2, dim=tuple(range(1, pred_ft.ndim)))
    target_enstrophy = 0.5 * torch.sum(K2 * torch.abs(target_ft)**2, dim=tuple(range(1, target_ft.ndim)))
    
    return torch.abs(pred_enstrophy - target_enstrophy) / (target_enstrophy + 1e-8)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error."""
    return F.mse_loss(pred, target)


def max_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Maximum absolute error."""
    return torch.max(torch.abs(pred - target).flatten(1), dim=1)[0]


def correlation_coefficient(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pearson correlation coefficient per sample."""
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    
    pred_mean = pred_flat.mean(1, keepdim=True)
    target_mean = target_flat.mean(1, keepdim=True)
    
    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean
    
    numerator = (pred_centered * target_centered).sum(1)
    denominator = torch.sqrt(
        (pred_centered**2).sum(1) * (target_centered**2).sum(1)
    )
    
    return numerator / (denominator + 1e-8)


class MetricsTracker:
    """Track and aggregate metrics during evaluation."""
    
    def __init__(self, metrics: List[Callable]):
        self.metrics = {m.__name__: [] for m in metrics}
        self.metric_fns = {m.__name__: m for m in metrics}
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        for name, fn in self.metric_fns.items():
            val = fn(pred, target)
            if isinstance(val, torch.Tensor):
                if val.ndim == 0:
                    self.metrics[name].append(val.item())
                else:
                    self.metrics[name].extend(val.cpu().tolist())
    
    def compute(self) -> Dict[str, float]:
        results = {}
        for name, values in self.metrics.items():
            if values:
                results[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        return results
    
    def reset(self):
        for name in self.metrics:
            self.metrics[name] = []


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    metrics: Optional[List[Callable]] = None
) -> Dict[str, Dict]:
    """
    Evaluate model on dataset.
    
    Args:
        model: Neural operator model
        dataloader: Data loader with (input, target) pairs
        device: Device to run on
        metrics: List of metric functions
    
    Returns:
        Dictionary of metric statistics
    """
    if metrics is None:
        metrics = [relative_l2_error, relative_l1_error, mse_loss, max_error, correlation_coefficient]
    
    tracker = MetricsTracker(metrics)
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch['input'].to(device)
                y = batch['output'].to(device)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
            
            pred = model(x)
            tracker.update(pred, y)
    
    return tracker.compute()


def zero_shot_superresolution(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    target_resolution: int,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate zero-shot super-resolution capability.
    
    Trains on low-res, evaluates on high-res without retraining.
    """
    model.eval()
    model.to(device)
    
    errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch['input'].to(device)
                y = batch['output'].to(device)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
            
            # Model trained on low-res, but we evaluate on high-res
            # Need to interpolate input to model's expected resolution
            # This is the zero-shot test
            pred = model(x)
            
            # If pred is low-res, interpolate to target
            if pred.shape[-1] != target_resolution:
                pred = F.interpolate(
                    pred.unsqueeze(1) if pred.ndim == 3 else pred,
                    size=target_resolution,
                    mode='bilinear' if pred.ndim == 4 else 'linear',
                    align_corners=False
                )
                if pred.ndim == 4:
                    pred = pred.squeeze(1)
            
            error = relative_l2_error(pred, y)
            errors.extend(error.cpu().tolist())
    
    return {
        'mean_rel_l2': np.mean(errors),
        'std_rel_l2': np.std(errors),
        'median_rel_l2': np.median(errors)
    }


def domain_adaptation_evaluation(
    model: torch.nn.Module,
    source_dataloader: torch.utils.data.DataLoader,
    target_dataloader: torch.utils.data.DataLoader,
    adaptation_steps: int = 10,
    lr: float = 1e-4,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate few-shot domain adaptation.
    
    Fine-tunes model on target domain for a few steps.
    """
    model.train()
    model.to(device)
    
    # Copy model for adaptation
    adapted_model = torch.nn.Sequential(*list(model.children()))
    adapted_model.load_state_dict(model.state_dict())
    adapted_model.to(device)
    
    optimizer = torch.optim.Adam(adapted_model.parameters(), lr=lr)
    
    # Few-shot adaptation
    for step in range(adaptation_steps):
        for batch in target_dataloader:
            if isinstance(batch, dict):
                x = batch['input'].to(device)
                y = batch['output'].to(device)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
            
            pred = adapted_model(x)
            loss = F.mse_loss(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step >= adaptation_steps:
                break
    
    # Evaluate adapted model
    adapted_model.eval()
    errors = []
    
    with torch.no_grad():
        for batch in target_dataloader:
            if isinstance(batch, dict):
                x = batch['input'].to(device)
                y = batch['output'].to(device)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
            
            pred = adapted_model(x)
            error = relative_l2_error(pred, y)
            errors.extend(error.cpu().tolist())
    
    return {
        'adapted_mean_rel_l2': np.mean(errors),
        'adapted_std_rel_l2': np.std(errors),
        'adaptation_steps': adaptation_steps
    }


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute all standard metrics at once."""
    return {
        'rel_l2': relative_l2_error(pred, target).mean().item(),
        'rel_l1': relative_l1_error(pred, target).mean().item(),
        'mse': mse_loss(pred, target).item(),
        'max_error': max_error(pred, target).mean().item(),
        'correlation': correlation_coefficient(pred, target).mean().item(),
        'energy_error': energy_error(pred, target).mean().item(),
    }


__all__ = [
    'relative_l2_error', 'relative_l1_error', 'spectral_error',
    'spectral_convergence_rate', 'energy_error', 'enstrophy_error',
    'mse_loss', 'max_error', 'correlation_coefficient',
    'MetricsTracker', 'evaluate_model', 'zero_shot_superresolution',
    'domain_adaptation_evaluation', 'compute_all_metrics',
]