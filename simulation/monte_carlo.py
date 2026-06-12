"""
Simulări Monte Carlo pentru evaluarea performanțelor SL-GLRT și ML-GLRT.

Funcțiile principale:
- compute_pfa_curve : curba Pfa(T) pentru un detector
- compute_pd_curve  : curba Pd(SNR) pentru un detector la prag fix
- threshold_for_pfa : găsește pragul T care produce o Pfa fixată
"""

import numpy as np
from typing import Union

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

from ..detectors.sl_glrt import SLGLRT
from ..detectors.ml_glrt import MLGLRT
from .generate_data import simulate_h0, simulate_h1, simulate_h0_multilook, simulate_h1_multilook


def _statistic_batch_sl(detector: SLGLRT, X_batch: np.ndarray) -> np.ndarray:
    """
    Calculează statistica SL-GLRT pentru un batch de vectori X (shape: n_samples × N).
    Folosește vectorizarea pentru viteză.
    """
    # |A^H x|^2 pentru fiecare pixel
    proj_sq = np.abs(X_batch @ detector.A.conj()) ** 2  # (n_samples, Ms*Mv)
    x_norm_sq = np.real(np.sum(np.abs(X_batch) ** 2, axis=1))  # (n_samples,)
    x_norm_sq = np.maximum(x_norm_sq, 1e-30)
    max_proj = np.max(proj_sq, axis=1)
    return max_proj / x_norm_sq


def _statistic_batch_ml(detector: MLGLRT, X_looks_batch: np.ndarray) -> np.ndarray:
    """
    Calculează statistica ML-GLRT pentru un batch de seturi de lookuri.

    Parameters
    ----------
    X_looks_batch : np.ndarray, shape (n_samples, L, N)
    """
    n_samples, L, N = X_looks_batch.shape
    stats = np.zeros(n_samples)
    for i in range(n_samples):
        # Calcul Ĉ pentru fiecare realizare
        Xl = X_looks_batch[i]  # (L, N)
        C = (Xl.conj().T @ Xl) / L  # (N, N)
        gamma, _, _, _, _ = detector.statistic_from_covariance(C)
        stats[i] = gamma
    return stats


def compute_pfa_curve(detector: Union[SLGLRT, MLGLRT],
                      thresholds: np.ndarray,
                      n_samples: int = 100_000,
                      L: int = 1,
                      seed: int = 42,
                      verbose: bool = True) -> np.ndarray:
    """
    Calculează Pfa pentru fiecare prag din `thresholds`, prin simulare Monte Carlo
    sub ipoteza H0 (doar zgomot).

    Parameters
    ----------
    detector : SLGLRT sau MLGLRT
    thresholds : np.ndarray, valorile pragului T
    n_samples : int, numărul de realizări Monte Carlo
    L : int, numărul de lookuri (pentru ML-GLRT)
    seed : int
    verbose : bool

    Returns
    -------
    pfa : np.ndarray, shape (len(thresholds),)
    """
    rng = np.random.default_rng(seed)
    N = detector.geom.n_images

    if isinstance(detector, MLGLRT):
        # Generează n_samples × L × N
        if verbose:
            print(f"Generare {n_samples} realizări H0 cu L={L}...")
        X_looks = simulate_h0_multilook(N, L, n_samples=n_samples, rng=rng)
        if verbose:
            print("Calcul statistici ML-GLRT...")
        stats = _statistic_batch_ml(detector, X_looks)
    else:
        X = simulate_h0(N, n_samples=n_samples, rng=rng)
        stats = _statistic_batch_sl(detector, X)

    # Pentru fiecare prag T, Pfa = fracția de stats > T
    pfa = np.array([np.mean(stats > t) for t in thresholds])
    return pfa


def compute_pd_curve(detector: Union[SLGLRT, MLGLRT],
                     threshold: float,
                     snr_db_values: np.ndarray,
                     s_true: float = 0.0,
                     v_true: float = 0.0,
                     n_samples: int = 10_000,
                     L: int = 1,
                     seed: int = 42,
                     verbose: bool = True) -> np.ndarray:
    """
    Calculează probabilitatea de detecție Pd pentru o serie de valori SNR.

    Parameters
    ----------
    detector : SLGLRT sau MLGLRT
    threshold : float, pragul T fixat
    snr_db_values : np.ndarray, valorile SNR în dB
    s_true, v_true : float, parametrii reali ai dispersorului
    n_samples : int
    L : int
    seed : int
    verbose : bool

    Returns
    -------
    pd : np.ndarray, shape (len(snr_db_values),)
    """
    rng = np.random.default_rng(seed)
    pd = np.zeros(len(snr_db_values))

    iterator = enumerate(snr_db_values)
    if verbose:
        iterator = tqdm(list(iterator), desc=f"Pd vs SNR (L={L})")

    for k, snr in iterator:
        if isinstance(detector, MLGLRT):
            X_looks = simulate_h1_multilook(detector.geom, s_true, v_true,
                                            snr_db=snr, L=L,
                                            n_samples=n_samples, rng=rng)
            stats = _statistic_batch_ml(detector, X_looks)
        else:
            X = simulate_h1(detector.geom, s_true, v_true,
                            snr_db=snr, n_samples=n_samples, rng=rng)
            stats = _statistic_batch_sl(detector, X)

        pd[k] = np.mean(stats > threshold)

    return pd


def threshold_for_pfa(detector: Union[SLGLRT, MLGLRT],
                      target_pfa: float,
                      n_samples: int = 100_000,
                      L: int = 1,
                      seed: int = 42) -> float:
    """
    Găsește pragul T care produce probabilitatea de alarmă falsă țintă.

    Folosește metoda quantilei empirice peste un eșantion H0.

    Parameters
    ----------
    detector : SLGLRT sau MLGLRT
    target_pfa : float, valoarea dorită a Pfa (ex: 1e-4)
    n_samples : int, numărul de realizări (trebuie > 1/target_pfa)
    L : int
    seed : int

    Returns
    -------
    T : float
    """
    if n_samples < 10 / target_pfa:
        print(f"AVERTISMENT: pentru Pfa={target_pfa}, recomandat n_samples >= "
              f"{int(10 / target_pfa)} (acum: {n_samples})")

    rng = np.random.default_rng(seed)
    N = detector.geom.n_images

    if isinstance(detector, MLGLRT):
        X_looks = simulate_h0_multilook(N, L, n_samples=n_samples, rng=rng)
        stats = _statistic_batch_ml(detector, X_looks)
    else:
        X = simulate_h0(N, n_samples=n_samples, rng=rng)
        stats = _statistic_batch_sl(detector, X)

    # Pragul = (1-target_pfa) quantila a statisticilor sub H0
    T = np.quantile(stats, 1 - target_pfa)
    return float(T)
