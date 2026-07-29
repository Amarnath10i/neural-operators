"""
Evaluation Package.
"""
from .metrics import (
    relative_l2_error,
    relative_l1_error,
    spectral_error,
    spectral_convergence_rate,
    energy_error,
    enstrophy_error,
    mse_loss,
    max_error,
    correlation_coefficient,
    MetricsTracker,
    evaluate_model,
    zero_shot_superresolution,
    domain_adaptation_evaluation,
    compute_all_metrics,
)

__all__ = [
    'relative_l2_error', 'relative_l1_error', 'spectral_error',
    'spectral_convergence_rate', 'energy_error', 'enstrophy_error',
    'mse_loss', 'max_error', 'correlation_coefficient',
    'MetricsTracker', 'evaluate_model', 'zero_shot_superresolution',
    'domain_adaptation_evaluation', 'compute_all_metrics',
]