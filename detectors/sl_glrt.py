"""
Implementarea detectorului SL-GLRT (Single-Look GLRT) pentru detecția
dispersorilor stabili în stive multi-temporale SAR.

Statistica de test (formula 3.18 din lucrarea de disertație):

    γ_SL-GLRT = max_{s,v} |a^H(s,v) * x|^2 / ||x||^2    H1
                                                       ≷   T
                                                       H0

Referință principală:
    De Maio, A., Fornaro, G., Pauciullo, A. (2009).
    "Detection of Single Scatterers in Multidimensional SAR Imaging."
    IEEE Trans. Geosci. Remote Sens., vol. 47, no. 7, pp. 2284-2297.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .steering import SystemGeometry, steering_dictionary, build_search_grid


@dataclass
class DetectionResult:
    """Rezultatul aplicării detectorului pe un pixel."""
    detected: bool          # Decizie binară (True = H1, False = H0)
    statistic: float        # Valoarea statisticii de test
    threshold: float        # Pragul folosit
    s_est: float           # Estimarea elevației reziduale (m)
    v_est: float           # Estimarea vitezei (m/an); pentru cm/an, *100
    s_est_idx: int         # Indexul în grila s al maximului
    v_est_idx: int         # Indexul în grila v al maximului


class SLGLRT:
    """
    Detectorul Single-Look GLRT.

    Aplică formula (3.18) pixel cu pixel, fără mediere spațială.

    Exemplu
    -------
    >>> geom = SystemGeometry(wavelength=0.0555, slant_range=800e3,
    ...                      baselines_perp=bperp, baselines_temp=btemp)
    >>> detector = SLGLRT(geom)
    >>> result = detector.detect(x, threshold=0.5)
    >>> print(f"Detected: {result.detected}, s={result.s_est:.1f} m, "
    ...       f"v={result.v_est*100:.2f} cm/an")
    """

    def __init__(self, geom: SystemGeometry,
                 s_grid: Optional[np.ndarray] = None,
                 v_grid: Optional[np.ndarray] = None,
                 s_range: tuple = (-50.0, 50.0),
                 v_range: tuple = (-0.03, 0.03),
                 oversampling: int = 4):
        """
        Parameters
        ----------
        geom : SystemGeometry
            Geometria sistemului și a stivei.
        s_grid, v_grid : np.ndarray, opțional
            Grila de căutare explicită. Dacă None, se construiește automat.
        s_range, v_range : tuple
            Intervalele de căutare (folosite dacă grilele nu sunt date).
        oversampling : int
            Factor de oversampling față de rezoluția Rayleigh (default 4).
        """
        self.geom = geom

        if s_grid is None or v_grid is None:
            s_grid, v_grid = build_search_grid(
                geom, s_range=s_range, v_range=v_range,
                s_oversampling=oversampling, v_oversampling=oversampling
            )
        self.s_grid = np.asarray(s_grid)
        self.v_grid = np.asarray(v_grid)

        # Pre-calculează dicționarul de vectori de direcție - O(N * Ms * Mv)
        # operație efectuată o singură dată, apoi reutilizată pentru toți pixelii
        self.A = steering_dictionary(self.s_grid, self.v_grid, geom)
        self.Ms = len(self.s_grid)
        self.Mv = len(self.v_grid)

    def statistic(self, x: np.ndarray) -> tuple:
        """
        Calculează statistica SL-GLRT și estimarea parametrilor pentru
        un vector de date x.

        Parameters
        ----------
        x : np.ndarray, shape (N,)
            Vectorul complex de date.

        Returns
        -------
        gamma : float
            Valoarea maximă a statisticii γ ∈ [0, 1].
        s_est, v_est : float
            Estimările MLE ale parametrilor (m și m/an).
        s_idx, v_idx : int
            Indexii în grilă ai maximului.
        """
        # Numărător: |A^H x|^2 pentru toate punctele grilei
        # A.conj().T @ x are shape (Ms*Mv,)
        proj = self.A.conj().T @ x  # produsul scalar pentru fiecare coloană
        proj_sq = np.abs(proj) ** 2  # |proiecție|^2

        # Maximul peste grilă
        max_idx = np.argmax(proj_sq)
        gamma_num = proj_sq[max_idx]

        # Numitor: ||x||^2
        x_norm_sq = np.real(x.conj() @ x)
        if x_norm_sq < 1e-30:
            return 0.0, 0.0, 0.0, 0, 0

        gamma = gamma_num / x_norm_sq

        # Indexii s și v din indexul plat
        v_idx, s_idx = divmod(max_idx, self.Ms)
        # NOTĂ: ordinea depinde de cum a fost construit A
        # A are shape (N, Ms*Mv) cu reshape(N, Ms, Mv).reshape(N, Ms*Mv)
        # Deci index plat = i*Mv + j  unde i=s_idx, j=v_idx
        s_idx = max_idx // self.Mv
        v_idx = max_idx % self.Mv

        s_est = self.s_grid[s_idx]
        v_est = self.v_grid[v_idx]

        return gamma, s_est, v_est, s_idx, v_idx

    def detect(self, x: np.ndarray, threshold: float) -> DetectionResult:
        """
        Aplică detectorul pe un singur pixel.

        Parameters
        ----------
        x : np.ndarray, shape (N,)
            Vectorul de date complex.
        threshold : float
            Pragul de detecție T ∈ [0, 1].

        Returns
        -------
        result : DetectionResult
        """
        gamma, s_est, v_est, s_idx, v_idx = self.statistic(x)
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
        Aplică detectorul pe o stivă completă de pixeli.

        Parameters
        ----------
        X : np.ndarray, shape (H, W, N)
            Stiva de date complex: H × W pixeli, N achiziții.
        threshold : float
            Pragul T.
        verbose : bool
            Afișează progres.

        Returns
        -------
        results : dict cu cheile:
            'detected' : np.ndarray, shape (H, W), bool
            'statistic' : np.ndarray, shape (H, W), float
            's_est' : np.ndarray, shape (H, W), float (m)
            'v_est' : np.ndarray, shape (H, W), float (m/an)
        """
        H, W, N = X.shape
        if N != self.geom.n_images:
            raise ValueError(f"Numărul de imagini din X ({N}) nu corespunde "
                             f"cu cel din geom ({self.geom.n_images})")

        # Reshape la (H*W, N)
        X_flat = X.reshape(-1, N)

        # Procesare vectorizată cu produsul matriceal
        # proj = X @ A.conj()  (H*W, Ms*Mv)
        proj_sq = np.abs(X_flat @ self.A.conj()) ** 2  # (H*W, Ms*Mv)

        # Numitor: ||x||^2 per pixel
        x_norm_sq = np.real(np.sum(np.abs(X_flat) ** 2, axis=1))  # (H*W,)
        x_norm_sq = np.maximum(x_norm_sq, 1e-30)

        # Maximul peste grilă pentru fiecare pixel
        max_idx_flat = np.argmax(proj_sq, axis=1)  # (H*W,)
        max_val = proj_sq[np.arange(len(X_flat)), max_idx_flat]  # (H*W,)

        gamma = max_val / x_norm_sq  # statisticile

        s_idx = max_idx_flat // self.Mv
        v_idx = max_idx_flat % self.Mv

        return {
            'detected': (gamma > threshold).reshape(H, W),
            'statistic': gamma.reshape(H, W),
            's_est': self.s_grid[s_idx].reshape(H, W),
            'v_est': self.v_grid[v_idx].reshape(H, W),
        }

    @staticmethod
    def threshold_analytical(pfa: float, N: int) -> float:
        """
        Pragul T care produce o Pfa fixată, conform formulei analitice
        (3.27) din lucrare (aproximarea Beta(1, N-1)):

            Pfa = (1 - T)^(N-1)  =>  T = 1 - Pfa^(1/(N-1))

        ATENȚIE: aproximarea este valabilă pentru grile fine (Ms*Mv mare).
        Pentru valori exacte, folosește simulare Monte Carlo.
        """
        if N <= 1:
            raise ValueError("N trebuie să fie > 1")
        return 1.0 - pfa ** (1.0 / (N - 1))
