"""
Implementarea detectorului ML-GLRT (Multi-Look GLRT) pentru detecția
dispersorilor stabili în stive multi-temporale SAR.

Statistica de test (formula 3.22 din lucrarea de disertație):

    γ_ML-GLRT = max_{s,v} a^H(s,v) * Ĉ * a(s,v) / tr(Ĉ)    H1
                                                          ≷   T
                                                          H0

unde Ĉ = (1/L) Σ_l x_l x_l^H este matricea de covarianță eșantion
pe L lookuri (vecinătate spațială).

Pentru L = 1, ML-GLRT se reduce exact la SL-GLRT.

Referință principală:
    Pauciullo, A., Reale, D., Franzé, W., Fornaro, G. (2018).
    "Multi-Look in GLRT-Based Detection of Single and Double Persistent
    Scatterers." IEEE Trans. Geosci. Remote Sens., vol. 56, no. 9,
    pp. 5125-5137.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .steering import SystemGeometry, steering_dictionary, build_search_grid
from .sl_glrt import DetectionResult


class MLGLRT:
    """
    Detectorul Multi-Look GLRT.

    Aplică formula (3.22) folosind o vecinătate spațială de L pixeli
    pentru a estima matricea de covarianță eșantion.

    Strategia de multilook implementată: fereastră dreptunghiulară fixă
    (window_size × window_size). Pentru window_size=1, ML-GLRT se reduce
    la SL-GLRT.

    Exemplu
    -------
    >>> geom = SystemGeometry(...)
    >>> detector = MLGLRT(geom, window_size=3)  # 3x3 = 9 lookuri
    >>> results = detector.detect_stack(X, threshold=0.3)
    """

    def __init__(self, geom: SystemGeometry,
                 window_size: int = 3,
                 s_grid: Optional[np.ndarray] = None,
                 v_grid: Optional[np.ndarray] = None,
                 s_range: tuple = (-50.0, 50.0),
                 v_range: tuple = (-0.03, 0.03),
                 oversampling: int = 4):
        """
        Parameters
        ----------
        geom : SystemGeometry
        window_size : int
            Dimensiunea ferestrei de mediere (impară). 3 => L=9, 5 => L=25.
        s_grid, v_grid : np.ndarray, opțional
        s_range, v_range : tuple
        oversampling : int
        """
        if window_size % 2 == 0:
            raise ValueError("window_size trebuie să fie impară")
        self.geom = geom
        self.window_size = window_size
        self.L = window_size ** 2  # numărul total de lookuri
        self.half_window = window_size // 2

        if s_grid is None or v_grid is None:
            s_grid, v_grid = build_search_grid(
                geom, s_range=s_range, v_range=v_range,
                s_oversampling=oversampling, v_oversampling=oversampling
            )
        self.s_grid = np.asarray(s_grid)
        self.v_grid = np.asarray(v_grid)
        self.A = steering_dictionary(self.s_grid, self.v_grid, geom)
        self.Ms = len(self.s_grid)
        self.Mv = len(self.v_grid)

    def statistic_from_covariance(self, C: np.ndarray) -> tuple:
        """
        Calculează statistica ML-GLRT direct dintr-o matrice de covarianță Ĉ.

        Formula: γ = max_{s,v} a^H Ĉ a / tr(Ĉ)

        Parameters
        ----------
        C : np.ndarray, shape (N, N), complex
            Matricea de covarianță Ĉ.

        Returns
        -------
        gamma, s_est, v_est, s_idx, v_idx
        """
        # Numărător: pentru fiecare coloană a din A, calculează a^H C a
        # CA = C @ A  are shape (N, Ms*Mv)
        # a^H C a = sum_n A[n,k].conj() * (C @ A)[n,k] over n axis
        CA = C @ self.A                                    # (N, Ms*Mv)
        quad_form = np.real(np.sum(self.A.conj() * CA, axis=0))  # (Ms*Mv,)

        # Numitor: tr(Ĉ)
        trace_C = np.real(np.trace(C))
        if trace_C < 1e-30:
            return 0.0, 0.0, 0.0, 0, 0

        # Maximul peste grilă
        max_idx = int(np.argmax(quad_form))
        gamma = quad_form[max_idx] / trace_C

        s_idx = max_idx // self.Mv
        v_idx = max_idx % self.Mv
        return gamma, self.s_grid[s_idx], self.v_grid[v_idx], s_idx, v_idx

    def statistic(self, X_looks: np.ndarray) -> tuple:
        """
        Calculează statistica ML-GLRT din L vectori de date (lookuri).

        Parameters
        ----------
        X_looks : np.ndarray, shape (L, N)
            Cei L vectori de date din vecinătatea pixelului central.

        Returns
        -------
        gamma, s_est, v_est, s_idx, v_idx
        """
        L = X_looks.shape[0]
        # Matricea de covarianță eșantion: Ĉ = (1/L) Σ_l x_l x_l^H
        C_hat = (X_looks.conj().T @ X_looks) / L  # (N, N)
        return self.statistic_from_covariance(C_hat)

    def detect(self, X_looks: np.ndarray, threshold: float) -> DetectionResult:
        """
        Aplică detectorul pe o singură vecinătate (L lookuri).

        Parameters
        ----------
        X_looks : np.ndarray, shape (L, N)
        threshold : float

        Returns
        -------
        DetectionResult
        """
        gamma, s_est, v_est, s_idx, v_idx = self.statistic(X_looks)
        return DetectionResult(
            detected=(gamma > threshold),
            statistic=gamma,
            threshold=threshold,
            s_est=s_est, v_est=v_est,
            s_est_idx=s_idx, v_est_idx=v_idx
        )

    def detect_stack(self, X: np.ndarray, threshold: float,
                     verbose: bool = False) -> dict:
        """
        Aplică detectorul pe o stivă completă (H × W × N) cu fereastră glisantă.

        Parameters
        ----------
        X : np.ndarray, shape (H, W, N)
        threshold : float
        verbose : bool

        Returns
        -------
        results : dict (vezi SLGLRT.detect_stack)
        """
        H, W, N = X.shape
        if N != self.geom.n_images:
            raise ValueError("dimensiunea N nu corespunde")

        # Rezultate
        detected = np.zeros((H, W), dtype=bool)
        statistic = np.zeros((H, W), dtype=np.float64)
        s_est = np.zeros((H, W), dtype=np.float64)
        v_est = np.zeros((H, W), dtype=np.float64)

        hw = self.half_window
        # Procesare cu fereastră glisantă (pixelii de margine sunt omiși)
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(hw, H - hw), desc=f"ML-GLRT L={self.L}")
            except ImportError:
                iterator = range(hw, H - hw)
        else:
            iterator = range(hw, H - hw)

        for i in iterator:
            for j in range(hw, W - hw):
                # Extrage fereastra (window_size × window_size × N)
                window = X[i - hw:i + hw + 1, j - hw:j + hw + 1, :]
                # Reshape la (L, N)
                X_looks = window.reshape(self.L, N)
                # Aplică detectorul
                gamma, s, v, _, _ = self.statistic(X_looks)
                statistic[i, j] = gamma
                detected[i, j] = (gamma > threshold)
                s_est[i, j] = s
                v_est[i, j] = v

        return {
            'detected': detected,
            'statistic': statistic,
            's_est': s_est,
            'v_est': v_est,
        }
