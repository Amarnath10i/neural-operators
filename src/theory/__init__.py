"""
Theory Package - Operator Learning Theory.
"""
from .generalization import (
    rademacher_complexity_empirical,
    covering_number_sobolev,
    generalization_bound_rademacher,
    spectral_convergence_rate,
    lipschitz_constant_estimate,
    stability_analysis,
    discretization_invariance_test,
    fno_approximation_bound,
    deeponet_approximation_bound,
)

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
]