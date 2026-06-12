"""
Modul pentru generarea de date sintetice pentru validare prin simulări Monte Carlo.

Generează vectori de date conform modelului din formula (3.1):

    x = γ ⊙ a(s, v) + w

unde:
    - γ ~ CN(0, σ²_γ * I_N)  sub H1, dispersor persistent
    - w ~ CN(0, σ²_w * I_N)  zgomot Gaussian complex circular
    - a(s, v) = vectorul de direcție

Sub H0, doar zgomot: x = w.

Pentru ML-GLRT, se generează L lookuri independente, fiecare cu
γ_l independent (model fluctuant).
"""

import numpy as np
from typing import Optional
from ..detectors.steering import SystemGeometry, steering_vector


def complex_gaussian(shape, sigma: float = 1.0, rng=None):
    """
    Generează un eșantion dintr-o distribuție CN(0, σ² I) cu varianța totală σ²
    (varianța per component reală/imaginară este σ²/2).
    """
    if rng is None:
        rng = np.random.default_rng()
    re = rng.normal(0.0, sigma / np.sqrt(2.0), shape)
    im = rng.normal(0.0, sigma / np.sqrt(2.0), shape)
    return re + 1j * im


def simulate_h0(N: int, sigma_w: float = 1.0, n_samples: int = 1, rng=None):
    """
    Generează vectori de date sub ipoteza H0 (doar zgomot).

    Parameters
    ----------
    N : int
        Lungimea vectorului (numărul de imagini).
    sigma_w : float
        Deviația standard a zgomotului (cu σ²_w = sigma_w²).
    n_samples : int
        Numărul de realizări.
    rng : numpy random Generator

    Returns
    -------
    x : np.ndarray, shape (n_samples, N) dacă n_samples > 1, altfel (N,)
    """
    if rng is None:
        rng = np.random.default_rng()
    x = complex_gaussian((n_samples, N), sigma=sigma_w, rng=rng)
    return x[0] if n_samples == 1 else x


def simulate_h1(geom: SystemGeometry, s_true: float, v_true: float,
                snr_db: float, n_samples: int = 1, rng=None,
                persistent: bool = True):
    """
    Generează vectori sub ipoteza H1 (dispersor + zgomot).

    Folosește convenția SNR = σ²_γ / σ²_w și σ²_w = 1 (normalizat).

    Modelul: x = γ * a(s,v) + w   (PS persistent: γ scalar complex)
            sau
             x = γ_vec ⊙ a(s,v) + w  (PS fluctuant: γ_vec ∈ CN(0, σ²_γ I))

    Pentru SL-GLRT, derivat sub H1: x ~ CN(0, σ²_γ a a^H + σ²_w I),
    care corespunde modelului persistent cu γ scalar.

    Parameters
    ----------
    geom : SystemGeometry
    s_true : float
        Elevația reală a dispersorului (m).
    v_true : float
        Viteza reală LOS (m/an).
    snr_db : float
        Raportul semnal-zgomot în dB.
    n_samples : int
        Număr de realizări (pentru Monte Carlo).
    rng : Generator
    persistent : bool
        True = PS persistent (γ scalar, aceeași fază/amplitudine la toate
                            epocile, modulat de a(s,v)).
        False = PS fluctuant (γ_vec, fază independentă per epocă).

    Returns
    -------
    x : np.ndarray, shape (n_samples, N) sau (N,)
    """
    if rng is None:
        rng = np.random.default_rng()
    N = geom.n_images

    # Convenții: σ_w² = 1, σ_γ² = 10^(snr/10)
    sigma_w = 1.0
    sigma_gamma = np.sqrt(10 ** (snr_db / 10.0))

    # Vectorul de direcție (același pentru toate realizările)
    a = steering_vector(s_true, v_true, geom)  # (N,)

    if persistent:
        # γ ~ CN(0, σ²_γ) - scalar complex, una per realizare
        gamma_scalar = complex_gaussian((n_samples,), sigma=sigma_gamma, rng=rng)
        # signal = γ * a (vectorul de direcție scalat de un complex)
        # Înmulțim cu sqrt(N) pentru că a are norma 1/sqrt(N) per element
        # Convenția SNR per pixel: amplitudinea semnalului per pixel are
        # E[|γ * a_n|²] = σ²_γ / N. Pentru ca SNR per pixel să fie σ²_γ / σ²_w,
        # nu folosim normalizarea 1/sqrt(N) când generăm semnalul.
        # Soluția: generăm cu vectorul de direcție NEnormalizat (faza pură)
        phase = -(4.0 * np.pi / geom.wavelength) * (
            geom.baselines_perp * s_true / geom.slant_range
            + geom.baselines_temp * v_true
        )
        a_unnorm = np.exp(1j * phase)  # (N,) cu |a_n| = 1
        signal = gamma_scalar[:, None] * a_unnorm[None, :]  # (n_samples, N)
    else:
        # Model fluctuant: γ_n ~ CN(0, σ²_γ) independent per epocă
        gamma_vec = complex_gaussian((n_samples, N), sigma=sigma_gamma, rng=rng)
        phase = -(4.0 * np.pi / geom.wavelength) * (
            geom.baselines_perp * s_true / geom.slant_range
            + geom.baselines_temp * v_true
        )
        a_unnorm = np.exp(1j * phase)
        signal = gamma_vec * a_unnorm[None, :]

    # Zgomot
    w = complex_gaussian((n_samples, N), sigma=sigma_w, rng=rng)

    x = signal + w
    return x[0] if n_samples == 1 else x


def simulate_h1_multilook(geom: SystemGeometry, s_true: float, v_true: float,
                          snr_db: float, L: int, n_samples: int = 1, rng=None):
    """
    Generează L lookuri sub H1 pentru testarea ML-GLRT.

    Modelul fluctuant din Pauciullo et al. 2018: fiecare look are propriul
    γ_l ~ CN(0, σ²_γ) (scalar complex) independent. Faza dispersorului este
    dată de vectorul de direcție a(s,v), constant peste lookuri (dispersor
    persistent în spațiu, fluctuant în timp/între lookuri).

    Returns
    -------
    X_looks : np.ndarray, shape (n_samples, L, N) sau (L, N)
    """
    if rng is None:
        rng = np.random.default_rng()
    N = geom.n_images

    sigma_w = 1.0
    sigma_gamma = np.sqrt(10 ** (snr_db / 10.0))

    # Vector de direcție nenormalizat
    phase = -(4.0 * np.pi / geom.wavelength) * (
        geom.baselines_perp * s_true / geom.slant_range
        + geom.baselines_temp * v_true
    )
    a_unnorm = np.exp(1j * phase)  # (N,) cu |a_n| = 1

    # Per realizare și per look: γ_l scalar complex
    gamma_scalar = complex_gaussian((n_samples, L), sigma=sigma_gamma, rng=rng)
    signal = gamma_scalar[..., None] * a_unnorm[None, None, :]  # (n_samples, L, N)

    w = complex_gaussian((n_samples, L, N), sigma=sigma_w, rng=rng)
    X_looks = signal + w
    return X_looks[0] if n_samples == 1 else X_looks


def simulate_h0_multilook(N: int, L: int, sigma_w: float = 1.0,
                          n_samples: int = 1, rng=None):
    """
    Generează L lookuri sub H0 (doar zgomot).

    Returns
    -------
    X_looks : np.ndarray, shape (n_samples, L, N) sau (L, N)
    """
    if rng is None:
        rng = np.random.default_rng()
    X = complex_gaussian((n_samples, L, N), sigma=sigma_w, rng=rng)
    return X[0] if n_samples == 1 else X
