"""Modul simulare Monte Carlo pentru validare."""
from .generate_data import (
    simulate_h0, simulate_h1, simulate_h0_multilook, simulate_h1_multilook,
    complex_gaussian
)
from .monte_carlo import compute_pfa_curve, compute_pd_curve, threshold_for_pfa

__all__ = [
    'simulate_h0', 'simulate_h1', 'simulate_h0_multilook', 'simulate_h1_multilook',
    'complex_gaussian',
    'compute_pfa_curve', 'compute_pd_curve', 'threshold_for_pfa',
]
