"""
Fourier Neural Operator (FNO) Architectures.

Implementation of FNO and variants for learning solution operators of PDEs.

References:
- Li et al., "Fourier Neural Operator for Parametric PDEs" (ICML 2021)
- Kovachki et al., "Neural Operator: Learning Maps Between Function Spaces" (2023)
- Wen et al., "F-FNO: Factorized Fourier Neural Operators" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Callable
import math

from .spectral_conv import (
    SpectralConv1d, SpectralConv2d, SpectralConv3d,
    FactorizedSpectralConv2d, LowRankSpectralConv2d,
    get_spectral_conv
)


class FNOBlock(nn.Module):
    """
    Single FNO Block: Spectral Conv + Pointwise Conv + Activation + Normalization.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, Tuple[int, ...]],
        dim: int = 2,
        activation: Callable = F.gelu,
        norm: bool = True,
        dropout: float = 0.0,
        spectral_conv_type: str = 'standard',  # 'standard', 'factorized', 'lowrank'
        rank: Optional[int] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.activation = activation
        
        # Spectral convolution
        if spectral_conv_type == 'standard':
            self.spectral_conv = get_spectral_conv(dim, in_channels, out_channels, modes)
        elif spectral_conv_type == 'factorized' and dim == 2:
            self.spectral_conv = FactorizedSpectralConv2d(in_channels, out_channels, modes)
        elif spectral_conv_type == 'lowrank' and dim == 2:
            self.spectral_conv = LowRankSpectralConv2d(
                in_channels, out_channels, modes, modes, rank
            )
        else:
            raise ValueError(f"Unknown spectral_conv_type: {spectral_conv_type}")
        
        # Pointwise convolution (local skip connection)
        if dim == 1:
            self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)
        elif dim == 2:
            self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)
        elif dim == 3:
            self.pointwise_conv = nn.Conv3d(in_channels, out_channels, 1)
        
        # Normalization
        if norm:
            if dim == 1:
                self.norm = nn.InstanceNorm1d(out_channels, affine=True)
            elif dim == 2:
                self.norm = nn.InstanceNorm2d(out_channels, affine=True)
            elif dim == 3:
                self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        else:
            self.norm = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectral path
        x_spectral = self.spectral_conv(x)
        
        # Local path
        x_local = self.pointwise_conv(x)
        
        # Combine
        x = x_spectral + x_local
        
        # Normalize and activate
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class FNO(nn.Module):
    """
    Fourier Neural Operator for learning solution operators of PDEs.
    
    Architecture:
    1. Lifting layer: project input to higher dimension
    2. N FNO blocks: spectral conv + pointwise conv + activation
    3. Projection layer: project back to output dimension
    
    Can handle 1D, 2D, and 3D problems.
    """
    
    def __init__(
        self,
        modes: Union[int, Tuple[int, ...]],
        width: int,
        n_layers: int = 4,
        in_channels: int = 3,
        out_channels: int = 1,
        dim: int = 2,
        lifting_channels: Optional[int] = None,
        projection_channels: Optional[int] = None,
        activation: Callable = F.gelu,
        norm: bool = True,
        dropout: float = 0.0,
        spectral_conv_type: str = 'standard',
        rank: Optional[int] = None,
        padding: int = 0,
        padding_mode: str = 'zeros'
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.padding = padding
        self.padding_mode = padding_mode
        
        # Lifting layer
        lift_c = lifting_channels or width * 2
        if dim == 1:
            self.lifting = nn.Conv1d(in_channels, lift_c, 1)
        elif dim == 2:
            self.lifting = nn.Conv2d(in_channels, lift_c, 1)
        elif dim == 3:
            self.lifting = nn.Conv3d(in_channels, lift_c, 1)
        
        # FNO blocks
        self.blocks = nn.ModuleList([
            FNOBlock(
                in_channels=lift_c if i == 0 else width,
                out_channels=width,
                modes=modes,
                dim=dim,
                activation=activation,
                norm=norm,
                dropout=dropout,
                spectral_conv_type=spectral_conv_type,
                rank=rank
            )
            for i in range(n_layers)
        ])
        
        # Projection layers
        proj_c = projection_channels or width * 2
        # Wrap functional activation in module for Sequential
        act_module = nn.GELU() if activation == F.gelu else nn.Identity()
        if dim == 1:
            self.projection = nn.Sequential(
                nn.Conv1d(width, proj_c, 1),
                act_module,
                nn.Conv1d(proj_c, out_channels, 1)
            )
        elif dim == 2:
            self.projection = nn.Sequential(
                nn.Conv2d(width, proj_c, 1),
                act_module,
                nn.Conv2d(proj_c, out_channels, 1)
            )
        elif dim == 3:
            self.projection = nn.Sequential(
                nn.Conv3d(width, proj_c, 1),
                act_module,
                nn.Conv3d(proj_c, out_channels, 1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, in_channels, *spatial_dims)
        Returns:
            Output of shape (batch, out_channels, *spatial_dims)
        """
        # Handle padding for non-periodic domains
        if self.padding > 0:
            if self.dim == 1:
                x = F.pad(x, (self.padding, self.padding), mode=self.padding_mode)
            elif self.dim == 2:
                x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 
                         mode=self.padding_mode)
            elif self.dim == 3:
                x = F.pad(x, (self.padding,)*6, mode=self.padding_mode)
        
        # Lifting
        x = self.lifting(x)
        
        # FNO blocks
        for block in self.blocks:
            x = block(x)
        
        # Projection
        x = self.projection(x)
        
        # Remove padding
        if self.padding > 0:
            if self.dim == 1:
                x = x[..., self.padding:-self.padding]
            elif self.dim == 2:
                x = x[..., self.padding:-self.padding, self.padding:-self.padding]
            elif self.dim == 3:
                x = x[..., self.padding:-self.padding, self.padding:-self.padding, 
                      self.padding:-self.padding]
        
        return x


class FNO1d(FNO):
    """1D FNO for 1D PDEs (Burgers, 1D Navier-Stokes, etc.)"""
    
    def __init__(self, modes: int, width: int = 64, **kwargs):
        super().__init__(modes=modes, width=width, dim=1, **kwargs)


class FNO2d(FNO):
    """2D FNO for 2D PDEs (Darcy, 2D Navier-Stokes, etc.)"""
    
    def __init__(self, modes: int, width: int = 64, **kwargs):
        super().__init__(modes=modes, width=width, dim=2, **kwargs)


class FNO3d(FNO):
    """3D FNO for 3D PDEs (3D Navier-Stokes, MHD, etc.)"""
    
    def __init__(self, modes: int, width: int = 32, **kwargs):
        super().__init__(modes=modes, width=width, dim=3, **kwargs)


class WNO(nn.Module):
    """
    Wavelet Neural Operator (WNO).
    
    Uses wavelet transform instead of Fourier transform for
    multi-resolution analysis and better handling of non-periodic boundaries.
    
    Reference: "Wavelet Neural Operators" (2022)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        n_layers: int = 4,
        wavelet: str = 'db4',
        levels: int = 3,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_layers = n_layers
        self.wavelet = wavelet
        self.levels = levels
        
        try:
            import pywt
            self.pywt = pywt
        except ImportError:
            raise ImportError("PyWavelets required for WNO: pip install pywt")
        
        # Lifting
        self.lifting = nn.Conv2d(in_channels, width, 1)
        
        # Wavelet layers (approximate with separable convolutions)
        self.wavelet_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1),
            ) for _ in range(n_layers)
        ])
        
        # Projection
        self.projection = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For now, use standard conv as wavelet approximation
        x = self.lifting(x)
        for block in self.wavelet_blocks:
            x = block(x) + x  # Residual
        x = self.projection(x)
        return x


class MWTBlock(nn.Module):
    """
    MultiWavelet Transform Block for MWT-NO.
    
    Reference: "Multipole Graph Neural Operator for PDEs" (2022)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        k: int = 3,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.k = k
        
        # Multipole expansion (simplified)
        self.multiplier = nn.Parameter(torch.randn(modes, modes, dtype=torch.cfloat))
        
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFT
        x_ft = torch.fft.rfft2(x, norm='ortho')
        
        # Multiply by multipole expansion
        b, c, h, w = x_ft.shape
        out_ft = torch.zeros_like(x_ft)
        
        m = min(self.modes, h, w)
        for i in range(m):
            for j in range(m):
                out_ft[:, :, i, j] = x_ft[:, :, i, j] * self.multiplier[i, j]
        
        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=x.shape[-2:], norm='ortho')
        
        # Local conv
        x = self.conv(x)
        x = self.activation(x)
        
        return x


class MWTNO(nn.Module):
    """Multipole Wavelet Transform Neural Operator."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        n_layers: int = 4,
        modes: int = 16,
        **kwargs
    ):
        super().__init__()
        
        self.lifting = nn.Conv2d(in_channels, width, 1)
        
        self.blocks = nn.ModuleList([
            MWTBlock(width, width, modes) for _ in range(n_layers)
        ])
        
        self.projection = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifting(x)
        for block in self.blocks:
            x = block(x) + x
        x = self.projection(x)
        return x


class HybridFNO(nn.Module):
    """
    Hybrid FNO combining spectral and local convolutions.
    
    Combines global spectral convolution with local message passing
    for better handling of complex geometries and non-periodic boundaries.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        n_layers: int = 4,
        modes: int = 16,
        local_kernel: int = 3,
        **kwargs
    ):
        super().__init__()
        
        self.lifting = nn.Conv2d(in_channels, width, 1)
        
        self.spectral_blocks = nn.ModuleList([
            SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)
        ])
        
        self.local_blocks = nn.ModuleList([
            nn.Conv2d(width, width, local_kernel, padding=local_kernel//2)
            for _ in range(n_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.InstanceNorm2d(width, affine=True) for _ in range(n_layers)
        ])
        
        self.activation = nn.GELU()
        
        self.projection = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifting(x)
        
        for spectral_conv, local_conv, norm in zip(
            self.spectral_blocks, self.local_blocks, self.norms
        ):
            x_spectral = spectral_conv(x)
            x_local = local_conv(x)
            x = norm(x_spectral + x_local)
            x = self.activation(x)
        
        x = self.projection(x)
        return x


# Factory functions
def get_fno_model(
    model_type: str,
    in_channels: int,
    out_channels: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create FNO variants.
    
    Args:
        model_type: 'fno1d', 'fno2d', 'fno3d', 'wno', 'mwtno', 'hybrid'
        in_channels: Input channels
        out_channels: Output channels
        **kwargs: Additional arguments
    
    Returns:
        Neural operator model
    """
    model_type = model_type.lower()
    
    if model_type == 'fno1d':
        return FNO1d(in_channels=in_channels, out_channels=out_channels, **kwargs)
    elif model_type == 'fno2d':
        return FNO2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
    elif model_type == 'fno3d':
        return FNO3d(in_channels=in_channels, out_channels=out_channels, **kwargs)
    elif model_type == 'wno':
        return WNO(in_channels=in_channels, out_channels=out_channels, **kwargs)
    elif model_type == 'mwtno':
        return MWTNO(in_channels=in_channels, out_channels=out_channels, **kwargs)
    elif model_type == 'hybrid':
        return HybridFNO(in_channels=in_channels, out_channels=out_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    'FNOBlock',
    'FNO', 'FNO1d', 'FNO2d', 'FNO3d',
    'WNO', 'MWTNO', 'HybridFNO',
    'MWTBlock',
    'get_fno_model',
]