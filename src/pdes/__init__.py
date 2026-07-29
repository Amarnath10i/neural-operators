"""
PDE Benchmarks Package.
"""
from .navier_stokes import (
    SpectralSolver1D,
    SpectralSolver2D,
    generate_burgers_data,
    generate_navier_stokes_data,
    generate_darcy_data,
    load_dataset,
    PDEDataset,
    Normalize,
    ToTensor,
)

__all__ = [
    'SpectralSolver1D', 'SpectralSolver2D',
    'generate_burgers_data', 'generate_navier_stokes_data', 'generate_darcy_data',
    'load_dataset', 'PDEDataset',
    'Normalize', 'ToTensor',
]