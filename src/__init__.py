"""
Neural Operators Package.

A comprehensive framework for neural operator learning:
- Fourier Neural Operators (FNO, WNO, FFNO, LNO)
- DeepONet variants (Standard, Stacked, POD)
- Graph Neural Operators (GNO, MNO)
- PDE Benchmarks (Navier-Stokes, Darcy, Burgers, etc.)
- Training & Evaluation pipelines
- Operator Learning Theory (Generalization bounds, Spectral analysis)
"""

from .operators import (
    SpectralConv1d, SpectralConv2d, SpectralConv3d,
    FactorizedSpectralConv2d, LowRankSpectralConv2d,
    FNO, FNO1d, FNO2d, FNO3d,
    WNO, MWTNO, HybridFNO,
    DeepONet, StackedDeepONet, PODDeepONet,
    GNO, GNOBlock, MNO, MNOBlock,
    LNO, LNOBlock,
    FFNO, FFNOBlock,
    get_fno_model, get_deeponet,
)
from .pdes import (
    SpectralSolver1D, SpectralSolver2D,
    generate_burgers_data, generate_navier_stokes_data, generate_darcy_data,
    PDEDataset,
)
from .training import (
    OperatorTrainer,
    SpectralLoss,
    PDELoss,
    RelativeL2Loss,
    create_optimizer,
    create_scheduler,
    get_loss_fn,
)
from .evaluation import (
    relative_l2_error,
    evaluate_model,
    zero_shot_superresolution,
    domain_adaptation_evaluation,
    compute_all_metrics,
)
from .theory import (
    rademacher_complexity_empirical,
    generalization_bound_rademacher,
    lipschitz_constant_estimate,
    stability_analysis,
    fno_approximation_bound,
    deeponet_approximation_bound,
)

__version__ = '0.1.0'
__author__ = 'Amarnath'

__all__ = [
    # Operators
    'SpectralConv1d', 'SpectralConv2d', 'SpectralConv3d',
    'FactorizedSpectralConv2d', 'LowRankSpectralConv2d',
    'FNO', 'FNO1d', 'FNO2d', 'FNO3d',
    'WNO', 'MWTNO', 'HybridFNO',
    'DeepONet', 'StackedDeepONet', 'PODDeepONet',
    'GNO', 'GNOBlock', 'MNO', 'MNOBlock',
    'LNO', 'LNOBlock',
    'FFNO', 'FFNOBlock',
    'get_fno_model', 'get_deeponet',
    # PDEs
    'SpectralSolver1D', 'SpectralSolver2D',
    'generate_burgers_data', 'generate_navier_stokes_data', 'generate_darcy_data',
    'PDEDataset',
    # Training
    'OperatorTrainer', 'SpectralLoss', 'PDELoss', 'RelativeL2Loss',
    'create_optimizer', 'create_scheduler', 'get_loss_fn',
    # Evaluation
    'relative_l2_error', 'evaluate_model', 'zero_shot_superresolution',
    'domain_adaptation_evaluation', 'compute_all_metrics',
    # Theory
    'rademacher_complexity_empirical', 'generalization_bound_rademacher',
    'lipschitz_constant_estimate', 'stability_analysis',
    'fno_approximation_bound', 'deeponet_approximation_bound',
]