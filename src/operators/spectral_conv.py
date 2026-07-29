"""
Spectral Convolution Operations for Neural Operators.

Implements efficient FFT-based convolutions in Fourier space as used in
Fourier Neural Operators (FNO) and related architectures.

References:
- Li et al., "Fourier Neural Operator for Parametric PDEs" (2021)
- Kovachki et al., "Neural Operator: Learning Maps Between Function Spaces" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution using FFT.
    
    Performs convolution in Fourier space by:
    1. FFT of input
    2. Multiply by learnable weights in Fourier domain (truncated to modes)
    3. Inverse FFT
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Learnable complex weights in Fourier space
        # Shape: (in_channels, out_channels, modes)
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, in_channels, spatial_dim)
        Returns:
            Output tensor of shape (batch, out_channels, spatial_dim)
        """
        batch_size, _, n = x.shape
        
        # FFT
        x_ft = torch.fft.rfft(x, norm='ortho')  # (batch, in_channels, n//2+1)
        
        # Multiply by weights in Fourier space (only first `modes` modes)
        out_ft = torch.zeros(
            batch_size, self.out_channels, n // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Truncated multiplication
        modes = min(self.modes, n // 2 + 1)
        out_ft[:, :, :modes] = torch.einsum(
            'bix,iox->box', x_ft[:, :, :modes], self.weights[:, :, :modes]
        )
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=n, norm='ortho')
        
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1)
        
        return x


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution using FFT.
    
    Performs 2D convolution in Fourier space with mode truncation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        batch_size, _, h, w = x.shape
        
        # 2D FFT
        x_ft = torch.fft.rfft2(x, norm='ortho')  # (batch, in_channels, h, w//2+1)
        
        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batch_size, self.out_channels, h, w // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Multiply low-frequency modes
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)
        
        # Corner 1: low-low frequencies
        out_ft[:, :, :m1, :m2] = torch.einsum(
            'bixy,ioxy->boxy', x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        
        # Corner 2: high-low frequencies (conjugate symmetry)
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            'bixy,ioxy->boxy', x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )
        
        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(h, w), norm='ortho')
        
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        
        return x


class SpectralConv3d(nn.Module):
    """3D Spectral Convolution for 3D PDEs (e.g., 3D Navier-Stokes)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(
                in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat
            )
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, d, h, w = x.shape
        
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm='ortho')
        
        out_ft = torch.zeros(
            batch_size, self.out_channels, d, h, w // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        m1 = min(self.modes1, d)
        m2 = min(self.modes2, h)
        m3 = min(self.modes3, w // 2 + 1)
        
        out_ft[:, :, :m1, :m2, :m3] = torch.einsum(
            'bixyz,ioxyz->boxyz', 
            x_ft[:, :, :m1, :m2, :m3], 
            self.weights[:, :, :m1, :m2, :m3]
        )
        
        x = torch.fft.irfftn(out_ft, s=(d, h, w), dim=(-3, -2, -1), norm='ortho')
        
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1, 1)
        
        return x


class FactorizedSpectralConv2d(nn.Module):
    """
    Factorized Spectral Convolution for efficiency.
    
    Decomposes 2D spectral convolution into separable 1D operations,
    reducing parameters from O(modes^2) to O(modes).
    
    Reference: "Factorized Fourier Neural Operators" (2023)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        
        # Separable weights for x and y dimensions
        self.weights_x = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
        self.weights_y = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = x.shape
        
        # FFT along x dimension
        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')  # (b, c, h, w//2+1)
        
        # Apply x weights
        m = min(self.modes, w // 2 + 1)
        x_ft[:, :, :, :m] = torch.einsum(
            'bchx,ocx->bcho', x_ft[:, :, :, :m], self.weights_x[:, :, :m]
        )
        
        # FFT along y dimension
        x_ft = torch.fft.rfft(x_ft, dim=-2, norm='ortho')  # (b, c, h//2+1, w//2+1)
        
        # Apply y weights
        m = min(self.modes, h // 2 + 1)
        x_ft[:, :, :m, :] = torch.einsum(
            'bchy,ocy->bcho', x_ft[:, :, :m, :], self.weights_y[:, :, :m]
        )
        
        # Inverse FFT
        x = torch.fft.irfft(x_ft, dim=-2, norm='ortho')
        x = torch.fft.irfft(x, dim=-1, n=w, norm='ortho')
        
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        
        return x


class LowRankSpectralConv2d(nn.Module):
    """
    Low-Rank Spectral Convolution.
    
    Approximates spectral weights using low-rank factorization
    for parameter efficiency.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        rank: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.rank = rank
        
        scale = 1.0 / math.sqrt(in_channels * out_channels * rank)
        
        # Low-rank factors: (in_channels, rank), (rank, out_channels), (modes1, modes2)
        self.U = nn.Parameter(scale * torch.randn(in_channels, rank))
        self.V = nn.Parameter(scale * torch.randn(rank, out_channels))
        self.spatial = nn.Parameter(scale * torch.randn(rank, modes1, modes2, dtype=torch.cfloat))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = x.shape
        
        x_ft = torch.fft.rfft2(x, norm='ortho')
        
        out_ft = torch.zeros(
            batch_size, self.out_channels, h, w // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)
        
        # Low-rank multiplication: (b, c, m1, m2) @ (c, r) @ (r, m1, m2) @ (r, oc)
        # Compute spatial factors
        spatial_weights = torch.einsum('rxy,cr,ro->coxy', 
                                       self.spatial[:, :m1, :m2], 
                                       self.U, 
                                       self.V)
        
        out_ft[:, :, :m1, :m2] = torch.einsum(
            'bixy,coxy->boxy', x_ft[:, :, :m1, :m2], spatial_weights
        )
        
        x = torch.fft.irfft2(out_ft, s=(h, w), norm='ortho')
        
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        
        return x


def get_spectral_conv(
    dim: int,
    in_channels: int,
    out_channels: int,
    modes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to get appropriate spectral convolution.
    
    Args:
        dim: Spatial dimension (1, 2, or 3)
        in_channels: Input channels
        out_channels: Output channels
        modes: Number of Fourier modes (or tuple for multi-dim)
        **kwargs: Additional arguments
    
    Returns:
        Spectral convolution module
    """
    if dim == 1:
        return SpectralConv1d(in_channels, out_channels, modes, **kwargs)
    elif dim == 2:
        if isinstance(modes, int):
            return SpectralConv2d(in_channels, out_channels, modes, modes, **kwargs)
        else:
            return SpectralConv2d(in_channels, out_channels, modes[0], modes[1], **kwargs)
    elif dim == 3:
        if isinstance(modes, int):
            return SpectralConv3d(in_channels, out_channels, modes, modes, modes, **kwargs)
        else:
            return SpectralConv3d(in_channels, out_channels, *modes, **kwargs)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")


# Export all
__all__ = [
    'SpectralConv1d',
    'SpectralConv2d', 
    'SpectralConv3d',
    'FactorizedSpectralConv2d',
    'LowRankSpectralConv2d',
    'get_spectral_conv',
]