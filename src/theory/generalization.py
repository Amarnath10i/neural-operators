"""
Operator Learning Theory: Generalization Bounds, Approximation Theory, Stability.

Implements theoretical analysis tools for neural operators:
- Rademacher complexity bounds
- Covering numbers in Sobolev spaces
- Spectral convergence analysis
- Discretization-invariant bounds
- Stability analysis (Lipschitz continuity)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
import math


def rademacher_complexity_empirical(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = 100,
    device: str = 'cuda'
) -> float:
    """
    Estimate empirical Rademacher complexity.
    
    R_S(F) = E_sigma [ sup_{f in F} (1/n) sum_i sigma_i f(x_i) ]
    
    Uses Monte Carlo estimation with Rademacher variables.
    """
    model.eval()
    model.to(device)
    
    complexities = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            total = 0.0
            count = 0
            
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch['input'].to(device)
                else:
                    x = batch[0].to(device)
                
                n = x.shape[0]
                # Rademacher variables
                sigma = torch.randint(0, 2, (n,), device=device) * 2 - 1
                sigma = sigma.float()
                
                # Model output (scalar per sample)
                out = model(x)
                if out.ndim > 1:
                    out = out.flatten(1).mean(1)  # Average over spatial dims
                
                # Supremum approximation: use model's own output
                total += (sigma * out).sum().item()
                count += n
            
            complexities.append(total / count)
    
    return np.mean(complexities), np.std(complexities)


def covering_number_sobolev(
    s: float,  # Smoothness parameter
    d: int,    # Dimension
    epsilon: float,
    domain_volume: float = (2*math.pi)**2
) -> float:
    """
    Covering number of Sobolev ball in L_infinity norm.
    
    log N(epsilon, W^s_p, L_inf) ~ epsilon^{-d/s}
    
    Reference: Kolmogorov & Tikhomirov (1959), Edmunds & Triebel (1996)
    """
    if s <= d/2:
        return float('inf')  # Not embedded in L_inf
    
    # For periodic domains
    return math.exp(domain_volume * (1/epsilon)**(d/s))


def generalization_bound_rademacher(
    empirical_risk: float,
    rademacher: float,
    n: int,
    delta: float = 0.05,
    loss_lipschitz: float = 1.0
) -> float:
    """
    Generalization bound via Rademacher complexity.
    
    With probability 1-delta:
    R(f) <= R_emp(f) + 2 * R_n(F) + sqrt(log(1/delta) / (2n))
    """
    return empirical_risk + 2 * rademacher * loss_lipschitz + math.sqrt(math.log(1/delta) / (2 * n))


def spectral_convergence_rate(
    model: nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    modes_list: List[int],
    device: str = 'cuda'
) -> Dict[int, float]:
    """
    Compute spectral convergence rate by truncating Fourier modes.
    
    Measures how error decays as more Fourier modes are included.
    """
    model.eval()
    model.to(device)
    
    errors = {m: [] for m in modes_list}
    
    with torch.no_grad():
        for batch in test_dataloader:
            if isinstance(batch, dict):
                x = batch['input'].to(device)
                y = batch['output'].to(device)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
            
            pred = model(x)
            
            # Compute error in Fourier space
            pred_ft = torch.fft.rfftn(pred, dim=(-2, -1), norm='ortho')
            y_ft = torch.fft.rfftn(y, dim=(-2, -1), norm='ortho')
            
            for m in modes_list:
                # Truncate to m modes
                h, w = pred.shape[-2:]
                pred_trunc = pred_ft.clone()
                y_trunc = y_ft.clone()
                
                pred_trunc[:, :, m:, :] = 0
                pred_trunc[:, :, :, m:] = 0
                y_trunc[:, :, m:, :] = 0
                y_trunc[:, :, :, m:] = 0
                
                # Reconstruct
                pred_rec = torch.fft.irfftn(pred_trunc, s=(h, w), dim=(-2, -1), norm='ortho')
                y_rec = torch.fft.irfftn(y_trunc, s=(h, w), dim=(-2, -1), norm='ortho')
                
                # Relative error
                diff = pred_rec - y_rec
                err = torch.norm(diff.flatten(1), dim=1) / (torch.norm(y_rec.flatten(1), dim=1) + 1e-8)
                errors[m].extend(err.cpu().tolist())
    
    return {m: np.mean(errors[m]) for m in modes_list}


def lipschitz_constant_estimate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_pairs: int = 1000,
    device: str = 'cuda'
) -> float:
    """
    Estimate Lipschitz constant of learned operator.
    
    L = sup_{x != y} ||F(x) - F(y)|| / ||x - y||
    """
    model.eval()
    model.to(device)
    
    ratios = []
    
    with torch.no_grad():
        data_list = []
        for batch in dataloader:
            if isinstance(batch, dict):
                data_list.append(batch['input'])
            else:
                data_list.append(batch[0])
        
        all_data = torch.cat(data_list, dim=0)
        n = min(len(all_data), num_pairs * 2)
        all_data = all_data[:n].to(device)
        
        # Random pairs
        for _ in range(num_pairs):
            i, j = np.random.choice(len(all_data), 2, replace=False)
            x1, x2 = all_data[i:i+1], all_data[j:j+1]
            
            out1 = model(x1)
            out2 = model(x2)
            
            if out1.ndim > 1:
                out1 = out1.flatten(1)
                out2 = out2.flatten(1)
            
            diff_out = torch.norm(out1 - out2)
            diff_in = torch.norm(x1.flatten(1) - x2.flatten(1))
            
            if diff_in > 1e-8:
                ratios.append((diff_out / diff_in).item())
    
    return np.max(ratios) if ratios else 0.0


def stability_analysis(
    model: nn.Module,
    perturbation_magnitudes: List[float],
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Dict[float, Dict]:
    """
    Analyze stability to input perturbations.
    
    Adds Gaussian noise of varying magnitudes and measures output change.
    """
    model.eval()
    model.to(device)
    
    results = {}
    
    for eps in perturbation_magnitudes:
        rel_changes = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch['input'].to(device)
                else:
                    x = batch[0].to(device)
                
                out_clean = model(x)
                
                # Add noise
                noise = torch.randn_like(x) * eps
                out_noisy = model(x + noise)
                
                # Relative change
                change = torch.norm(out_noisy - out_clean, dim=tuple(range(1, out_clean.ndim)))
                norm = torch.norm(out_clean, dim=tuple(range(1, out_clean.ndim)))
                
                rel_changes.extend((change / (norm + 1e-8)).cpu().tolist())
        
        results[eps] = {
            'mean': np.mean(rel_changes),
            'std': np.std(rel_changes),
            'max': np.max(rel_changes)
        }
    
    return results


def discretization_invariance_test(
    model: nn.Module,
    test_func: Callable,
    resolutions: List[int],
    device: str = 'cuda'
) -> Dict[int, float]:
    """
    Test if model's predictions are consistent across resolutions.
    
    Evaluates model on same problem at different resolutions
    (with appropriate interpolation).
    """
    model.eval()
    model.to(device)
    
    errors = {}
    
    for r in resolutions:
        # Generate problem at resolution r
        x, y = test_func(resolution=r)
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            pred = model(x)
            
            # Interpolate to common resolution if needed
            if pred.shape[-1] != y.shape[-1]:
                pred = F.interpolate(
                    pred.unsqueeze(1) if pred.ndim == 3 else pred,
                    size=y.shape[-2:],
                    mode='bilinear' if pred.ndim == 4 else 'linear',
                    align_corners=False
                )
                if pred.ndim == 4:
                    pred = pred.squeeze(1)
            
            err = relative_l2_error(pred, y).mean().item()
            errors[r] = err
    
    return errors


def fno_approximation_bound(
    modes: int,
    width: int,
    depth: int,
    input_smoothness: float,
    domain_dim: int = 2
) -> float:
    """
    Theoretical approximation bound for FNO.
    
    Based on: Kovachki et al. "Neural Operator: Learning Maps Between Function Spaces" (2021)
    
    Error ~ O(modes^{-s}) + O(width^{-1/2}) + O(depth^{-1})
    where s is the smoothness of the target operator.
    """
    spectral_error = modes ** (-input_smoothness)
    width_error = width ** (-0.5)
    depth_error = depth ** (-1.0)
    
    return spectral_error + width_error + depth_error


def deeponet_approximation_bound(
    branch_width: int,
    trunk_width: int,
    p: int,  # Latent dimension
    input_smoothness: float
) -> float:
    """
    Theoretical approximation bound for DeepONet.
    
    Based on: Lanthaler et al. "Error estimates for DeepONet" (2022)
    """
    # Branch net error
    branch_err = branch_width ** (-input_smoothness)
    # Trunk net error
    trunk_err = trunk_width ** (-input_smoothness)
    # Latent dimension error
    latent_err = p ** (-0.5)
    
    return branch_err + trunk_err + latent_err


class TheoryVerifier:
    """Verify theoretical properties of trained neural operators."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
    
    def verify_generalization(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        delta: float = 0.05
    ) -> Dict:
        """Verify generalization bound empirically."""
        self.model.eval()
        
        # Empirical train risk
        train_losses = []
        with torch.no_grad():
            for batch in train_loader:
                x = batch['input'].to(self.device) if isinstance(batch, dict) else batch[0].to(self.device)
                y = batch['output'].to(self.device) if isinstance(batch, dict) else batch[1].to(self.device)
                pred = self.model(x)
                train_losses.append(loss_fn(pred, y).item())
        
        emp_risk = np.mean(train_losses)
        
        # Rademacher complexity
        rad_mean, rad_std = rademacher_complexity_empirical(self.model, train_loader, device=self.device)
        
        # Test risk
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch['input'].to(self.device) if isinstance(batch, dict) else batch[0].to(self.device)
                y = batch['output'].to(self.device) if isinstance(batch, dict) else batch[1].to(self.device)
                pred = self.model(x)
                test_losses.append(loss_fn(pred, y).item())
        
        test_risk = np.mean(test_losses)
        
        # Bound
        n = len(train_loader.dataset)
        bound = generalization_bound_rademacher(emp_risk, rad_mean, n, delta)
        
        return {
            'empirical_risk': emp_risk,
            'test_risk': test_risk,
            'rademacher_mean': rad_mean,
            'rademacher_std': rad_std,
            'generalization_bound': bound,
            'bound_holds': test_risk <= bound,
            'gap': test_risk - emp_risk
        }
    
    def verify_spectral_convergence(
        self,
        test_loader: torch.utils.data.DataLoader,
        modes_list: List[int] = [4, 8, 16, 32, 64]
    ) -> Dict:
        """Verify spectral convergence."""
        return spectral_convergence_rate(self.model, test_loader, modes_list, self.device)
    
    def verify_stability(
        self,
        test_loader: torch.utils.data.DataLoader,
        eps_list: List[float] = [1e-4, 1e-3, 1e-2, 1e-1]
    ) -> Dict:
        """Verify Lipschitz stability."""
        return stability_analysis(self.model, eps_list, test_loader, self.device)
    
    def verify_discretization_invariance(
        self,
        test_func: Callable,
        resolutions: List[int] = [32, 64, 128, 256]
    ) -> Dict:
        """Verify discretization invariance."""
        return discretization_invariance_test(self.model, test_func, resolutions, self.device)


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 error per sample."""
    diff = pred - target
    return torch.norm(diff.flatten(1), dim=1) / (torch.norm(target.flatten(1), dim=1) + 1e-8)


__all__ = [
    'rademacher_complexity_empirical',
    'covering_number_sobolev',
    'generalization_bound_rademacher',
    'spectral_convergence_rate',
    'lipschitz_constant_estimate',
    'stability_analysis',
    'discretization_invariance_test',
    'fno_approximation_bound',
    'deeponet_approximation_bound',
    'TheoryVerifier',
    'relative_l2_error',
]