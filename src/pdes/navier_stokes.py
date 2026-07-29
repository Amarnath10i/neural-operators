"""
PDE Benchmarks and Data Generation for Neural Operators.

Generates training/test data for standard PDE benchmarks:
- Navier-Stokes (2D/3D)
- Darcy Flow
- Burgers Equation
- Heat Equation
- Wave Equation
- Reaction-Diffusion
- Euler Equations
- MHD

Uses spectral methods for high-accuracy ground truth.
"""

import torch
import torch.nn.functional as F
import numpy as np
import h5py
import os
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path
import math
from tqdm import tqdm


class SpectralSolver1D:
    """1D Spectral solver using FFT."""
    
    def __init__(self, n: int, L: float = 2*math.pi, device: str = 'cpu'):
        self.n = n
        self.L = L
        self.dx = L / n
        self.device = device
        
        # Wavenumbers for rfft (n//2 + 1 modes)
        k = torch.fft.rfftfreq(n, d=self.dx) * 2 * math.pi
        self.k = k.to(device)
        self.k2 = (k ** 2).to(device)
        self.k4 = (k ** 4).to(device)
    
    def solve_burgers(
        self,
        u0: torch.Tensor,
        nu: float,
        T: float,
        dt: float,
        forcing: Optional[torch.Tensor] = None,
        save_every: int = 1
    ) -> torch.Tensor:
        """
        Solve 1D Burgers equation: u_t + u u_x = nu u_xx + f
        using integrating factor method.
        """
        n_steps = int(T / dt)
        u = u0.clone()
        
        # Integrating factor for diffusion
        diff_factor = torch.exp(-nu * self.k2 * dt)
        
        n_save = n_steps // save_every + 1
        trajectory = torch.zeros(n_save, *u.shape, device=self.device, dtype=u.dtype)
        trajectory[0] = u
        save_idx = 1
        
        for step in range(1, n_steps + 1):
            # Nonlinear term in physical space
            u_hat = torch.fft.rfft(u)
            u_x = torch.fft.irfft(1j * self.k * u_hat)
            nonlinear = -u * u_x
            
            # Add forcing
            if forcing is not None:
                nonlinear = nonlinear + forcing
            
            # Time step in Fourier space
            nonlinear_hat = torch.fft.rfft(nonlinear)
            
            u_hat = diff_factor * u_hat + dt * diff_factor * nonlinear_hat
            u = torch.fft.irfft(u_hat, n=self.n)
            
            if step % save_every == 0:
                trajectory[save_idx] = u
                save_idx += 1
        
        return trajectory[:save_idx]
    
    def solve_heat(
        self,
        u0: torch.Tensor,
        alpha: float,
        T: float,
        dt: float,
        forcing: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Solve 1D Heat equation: u_t = alpha u_xx + f"""
        n_steps = int(T / dt)
        u = u0.clone()
        
        diff_factor = torch.exp(-alpha * self.k2 * dt)
        
        for _ in range(n_steps):
            u_hat = torch.fft.rfft(u)
            
            if forcing is not None:
                f_hat = torch.fft.rfft(forcing)
                u_hat = diff_factor * u_hat + dt * diff_factor * f_hat
            else:
                u_hat = diff_factor * u_hat
            
            u = torch.fft.irfft(u_hat, n=self.n)
        
        return u


class SpectralSolver2D:
    """2D Spectral solver using FFT."""
    
    def __init__(self, n: int, L: float = 2*math.pi, device: str = 'cpu'):
        self.n = n
        self.L = L
        self.dx = L / n
        self.device = device
        
        # 2D Wavenumbers
        kx = torch.fft.fftfreq(n, d=self.dx) * 2 * math.pi
        ky = torch.fft.rfftfreq(n, d=self.dx) * 2 * math.pi
        
        self.kx = kx.to(device)
        self.ky = ky.to(device)
        
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        self.KX = KX.to(device)
        self.KY = KY.to(device)
        
        self.K2 = (KX**2 + KY**2).to(device)
        self.K4 = (self.K2**2).to(device)
        
        # Laplacian inverse (for Poisson)
        self.laplacian_inv = torch.zeros_like(self.K2)
        self.laplacian_inv[1:, :] = -1.0 / self.K2[1:, :]
        self.laplacian_inv[0, 1:] = -1.0 / self.K2[0, 1:]
        self.laplacian_inv = self.laplacian_inv.to(device)
    
    def solve_navier_stokes_vorticity(
        self,
        w0: torch.Tensor,
        nu: float,
        T: float,
        dt: float,
        forcing: Optional[torch.Tensor] = None,
        save_every: int = 1
    ) -> torch.Tensor:
        """
        Solve 2D Navier-Stokes in vorticity form:
        w_t + u·∇w = nu ∇²w + f
        where u = ∇^⊥ ψ, ∇²ψ = w
        """
        n_steps = int(T / dt)
        w = w0.clone()
        
        n_save = n_steps // save_every + 1
        trajectory = torch.zeros(n_save, *w.shape, device=self.device, dtype=w.dtype)
        trajectory[0] = w
        save_idx = 1
        
        diff_factor = torch.exp(-nu * self.K2 * dt)
        
        for step in range(1, n_steps + 1):
            # Compute velocity from vorticity (stream function)
            psi_hat = torch.fft.rfft2(w) * self.laplacian_inv
            u_x_hat = 1j * self.KY * psi_hat
            u_y_hat = -1j * self.KX * psi_hat
            
            u_x = torch.fft.irfft2(u_x_hat, s=(self.n, self.n))
            u_y = torch.fft.irfft2(u_y_hat, s=(self.n, self.n))
            
            # Advection term: u·∇w
            w_x = torch.fft.irfft2(1j * self.KX * torch.fft.rfft2(w))
            w_y = torch.fft.irfft2(1j * self.KY * torch.fft.rfft2(w))
            
            advection = -(u_x * w_x + u_y * w_y)
            
            # Add forcing
            if forcing is not None:
                advection = advection + forcing
            
            # Time step in Fourier space
            w_hat = torch.fft.rfft2(w)
            adv_hat = torch.fft.rfft2(advection)
            
            w_hat = diff_factor * w_hat + dt * diff_factor * adv_hat
            w = torch.fft.irfft2(w_hat, s=(self.n, self.n))
            
            if step % save_every == 0:
                trajectory[save_idx] = w
                save_idx += 1
        
        return trajectory[:save_idx]
    
    def solve_darcy(
        self,
        a: torch.Tensor,
        f: torch.Tensor,
        max_iter: int = 50,
        tol: float = 1e-6
    ) -> torch.Tensor:
        """
        Solve Darcy flow: -∇·(a ∇u) = f
        Using Picard iteration with spectral method.
        """
        a = a.to(self.device)
        f = f.to(self.device)
        
        # Initial guess
        u = torch.zeros_like(a)
        
        f_hat = torch.fft.rfft2(f)
        
        for iteration in range(max_iter):
            u_hat = torch.fft.rfft2(u)
            u_x = torch.fft.irfft2(1j * self.KX * u_hat, s=(self.n, self.n))
            u_y = torch.fft.irfft2(1j * self.KY * u_hat, s=(self.n, self.n))
            
            # a * ∇u
            ax = a * u_x
            ay = a * u_y
            
            # ∇·(a ∇u)
            ax_hat = torch.fft.rfft2(ax)
            ay_hat = torch.fft.rfft2(ay)
            div = torch.fft.irfft2(1j * self.KX * ax_hat + 1j * self.KY * ay_hat, s=(self.n, self.n))
            
            # Residual: f + ∇·(a ∇u)
            residual = f + div
            residual_hat = torch.fft.rfft2(residual)
            
            # Update: u = u + (-∇²)^{-1} residual
            u_hat = residual_hat * self.laplacian_inv
            u_new = torch.fft.irfft2(u_hat, s=(self.n, self.n))
            
            # Check convergence
            diff = torch.norm(u_new - u) / (torch.norm(u) + 1e-8)
            u = u_new
            
            if diff < tol:
                break
        
        return u


def generate_burgers_data(
    n_samples: int = 1000,
    n: int = 1024,
    nu_range: Tuple[float, float] = (0.01, 0.1),
    T: float = 1.0,
    dt: float = 1e-3,
    save_every: int = 100,
    seed: int = 42,
    output_path: str = 'data/burgers.h5'
) -> Dict:
    """Generate 1D Burgers equation dataset with stable parameters."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    solver = SpectralSolver1D(n, device=device)
    
    inputs = []
    outputs = []
    nus = []
    
    print(f"Generating {n_samples} Burgers samples...")
    
    for i in tqdm(range(n_samples)):
        # Random initial condition (sum of sinusoids with controlled amplitude)
        n_modes = np.random.randint(3, 8)
        u0 = torch.zeros(n, device=device)
        for _ in range(n_modes):
            k = np.random.randint(1, n//4)
            a = np.random.randn() * 1.0
            b = np.random.randn() * 1.0
            phase = np.random.rand() * 2 * math.pi
            x = torch.linspace(0, 2*math.pi, n, device=device)
            u0 += a * torch.sin(k * x + phase) + b * torch.cos(k * x + phase)
        
        # Random viscosity - higher for stability
        nu = np.random.uniform(*nu_range)
        
        # Solve with NaN checking
        try:
            u_T = solver.solve_burgers(u0.unsqueeze(0), nu, T, dt)
            if torch.isnan(u_T).any():
                continue
            
            inputs.append(u0.cpu())
            outputs.append(u_T[-1].squeeze(0).cpu())
            nus.append(nu)
        except Exception:
            continue
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('input', data=torch.stack(inputs).numpy())
        f.create_dataset('output', data=torch.stack(outputs).numpy())
        f.create_dataset('nu', data=np.array(nus))
        f.attrs['n'] = n
        f.attrs['T'] = T
        f.attrs['dt'] = dt
        f.attrs['nu_range'] = nu_range
    
    print(f"Generated {len(inputs)} valid samples. Saved to {output_path}")
    return {'input': inputs, 'output': outputs, 'nu': nus}


def generate_navier_stokes_data(
    n_samples: int = 1000,
    n: int = 64,
    nu_range: Tuple[float, float] = (1e-4, 1e-3),
    T: float = 50.0,
    dt: float = 1e-3,
    save_every: int = 1000,
    forcing_type: str = 'kolmogorov',
    seed: int = 42,
    output_path: str = 'data/navier_stokes.h5'
) -> Dict:
    """Generate 2D Navier-Stokes dataset (vorticity formulation)."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    solver = SpectralSolver2D(n, device=device)
    
    # Forcing
    if forcing_type == 'kolmogorov':
        y = torch.linspace(0, 2*math.pi, n, device=device)
        forcing = 0.1 * torch.sin(4 * y).unsqueeze(0).repeat(n, 1)
    else:
        forcing = None
    
    inputs = []
    outputs = []
    nus = []
    trajectories = []
    
    print(f"Generating {n_samples} Navier-Stokes samples...")
    
    for i in tqdm(range(n_samples)):
        # Random initial vorticity
        n_modes = np.random.randint(5, 15)
        w0 = torch.zeros(n, n, device=device)
        
        for _ in range(n_modes):
            kx = np.random.randint(1, n//3)
            ky = np.random.randint(1, n//3)
            a = np.random.randn()
            phase = np.random.rand() * 2 * math.pi
            
            X, Y = torch.meshgrid(
                torch.linspace(0, 2*math.pi, n, device=device),
                torch.linspace(0, 2*math.pi, n, device=device),
                indexing='ij'
            )
            w0 += a * torch.sin(kx * X + ky * Y + phase)
        
        # Random viscosity
        nu = np.random.uniform(*nu_range)
        
        # Forcing
        if forcing_type == 'random':
            f = torch.randn(n, n, device=device) * 0.1
        else:
            f = forcing
        
        # Solve
        traj = solver.solve_navier_stokes_vorticity(
            w0.unsqueeze(0), nu, T, dt, f, save_every
        )
        
        inputs.append(w0.cpu())
        outputs.append(traj[-1].cpu())
        nus.append(nu)
        trajectories.append(traj.cpu())
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('input', data=torch.stack(inputs).numpy())
        f.create_dataset('output', data=torch.stack(outputs).numpy())
        f.create_dataset('trajectory', data=torch.stack(trajectories).numpy())
        f.create_dataset('nu', data=np.array(nus))
        f.attrs['n'] = n
        f.attrs['T'] = T
        f.attrs['dt'] = dt
        f.attrs['nu_range'] = nu_range
        f.attrs['forcing_type'] = forcing_type
    
    print(f"Saved to {output_path}")
    return {'input': inputs, 'output': outputs, 'nu': nus, 'trajectory': trajectories}


def generate_darcy_data(
    n_samples: int = 1000,
    n: int = 85,
    permeability_range: Tuple[float, float] = (0.1, 10.0),
    correlation_length: float = 0.1,
    seed: int = 42,
    output_path: str = 'data/darcy.h5'
) -> Dict:
    """Generate Darcy flow dataset with random permeability fields."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    solver = SpectralSolver2D(n, device=device)
    
    # Forcing (constant)
    f = torch.ones(n, n, device=device)
    
    inputs = []
    outputs = []
    
    print(f"Generating {n_samples} Darcy samples...")
    
    for i in tqdm(range(n_samples)):
        # Generate log-permeability field (Gaussian random field)
        # Using spectral method for correlation
        field = torch.randn(n, n, device=device)
        field_hat = torch.fft.rfft2(field)
        
        # Exponential covariance kernel in Fourier space
        k = torch.fft.fftfreq(n).to(device)
        KX, KY = torch.meshgrid(k, k[:n//2+1], indexing='ij')
        K2 = KX**2 + KY**2
        cov = torch.exp(-2 * math.pi * correlation_length * torch.sqrt(K2 + 1e-10))
        
        field_hat = field_hat * cov
        log_a = torch.fft.irfft2(field_hat, s=(n, n))
        
        # Normalize
        log_a = log_a - log_a.mean()
        log_a = log_a / log_a.std()
        
        # Map to permeability range
        a_min, a_max = permeability_range
        log_a_min, log_a_max = math.log(a_min), math.log(a_max)
        log_a = log_a * (log_a_max - log_a_min) / 2 + (log_a_max + log_a_min) / 2
        a = torch.exp(log_a)
        
        # Solve
        u = solver.solve_darcy(a, f)
        
        inputs.append(a.cpu())
        outputs.append(u.cpu())
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('input', data=torch.stack(inputs).numpy())
        f.create_dataset('output', data=torch.stack(outputs).numpy())
        f.attrs['n'] = n
        f.attrs['permeability_range'] = permeability_range
        f.attrs['correlation_length'] = correlation_length
    
    print(f"Saved to {output_path}")
    return {'input': inputs, 'output': outputs}


def load_dataset(path: str) -> Dict[str, torch.Tensor]:
    """Load dataset from HDF5."""
    with h5py.File(path, 'r') as f:
        data = {}
        for key in f.keys():
            data[key] = torch.from_numpy(f[key][:])
        return data


class PDEDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for PDE data."""
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        transform=None
    ):
        self.data = load_dataset(data_path)
        self.transform = transform
        
        n = len(self.data['input'])
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        if split == 'train':
            self.indices = slice(0, train_end)
        elif split == 'val':
            self.indices = slice(train_end, val_end)
        elif split == 'test':
            self.indices = slice(val_end, n)
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def __len__(self):
        if isinstance(self.indices, slice):
            return len(self.data['input'][self.indices])
        return len(self.indices)
    
    def __getitem__(self, idx):
        if isinstance(self.indices, slice):
            real_idx = idx + (self.indices.start or 0)
        else:
            real_idx = self.indices[idx]
        
        item = {
            'input': self.data['input'][real_idx].unsqueeze(0),  # Add channel dim
            'output': self.data['output'][real_idx].unsqueeze(0)  # Add channel dim
        }
        
        if 'nu' in self.data:
            item['nu'] = self.data['nu'][real_idx]
        
        if self.transform:
            item = self.transform(item)
        
        return item
    
    def loader(self, batch_size: int = 32, shuffle: bool = True, **kwargs):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)


# Normalization transforms
class Normalize:
    """Normalize input/output to zero mean, unit variance."""
    
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
    
    def __call__(self, item: Dict) -> Dict:
        item['input'] = (item['input'] - self.mean) / self.std
        item['output'] = (item['output'] - self.mean) / self.std
        return item


class ToTensor:
    """Ensure tensors."""
    
    def __call__(self, item: Dict) -> Dict:
        for k, v in item.items():
            if not isinstance(v, torch.Tensor):
                item[k] = torch.tensor(v, dtype=torch.float32)
        return item


__all__ = [
    'SpectralSolver1D', 'SpectralSolver2D',
    'generate_burgers_data', 'generate_navier_stokes_data', 'generate_darcy_data',
    'load_dataset', 'PDEDataset',
    'Normalize', 'ToTensor',
]