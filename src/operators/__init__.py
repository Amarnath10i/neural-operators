"""
Neural Operators - Operator Architectures.
"""

from .spectral_conv import (
    SpectralConv1d, SpectralConv2d, SpectralConv3d,
    FactorizedSpectralConv2d, LowRankSpectralConv2d,
    get_spectral_conv
)

from .fno import (
    FNOBlock, FNO, FNO1d, FNO2d, FNO3d,
    WNO, MWTNO, HybridFNO, MWTBlock,
    get_fno_model
)

from .deeponet import (
    DeepONet, StackedDeepONet, PODDeepONet,
    GNO, GNOBlock,
    MNO, MNOBlock,
    LNO, LNOBlock,
    FFNO, FFNOBlock,
    get_deeponet
)

__all__ = [
    'SpectralConv1d', 'SpectralConv2d', 'SpectralConv3d',
    'FactorizedSpectralConv2d', 'LowRankSpectralConv2d',
    'get_spectral_conv',
    'FNOBlock', 'FNO', 'FNO1d', 'FNO2d', 'FNO3d',
    'WNO', 'MWTNO', 'HybridFNO', 'MWTBlock',
    'get_fno_model',
    'DeepONet', 'StackedDeepONet', 'PODDeepONet',
    'GNO', 'GNOBlock',
    'MNO', 'MNOBlock',
    'LNO', 'LNOBlock',
    'FFNO', 'FFNOBlock',
    'get_deeponet',
]