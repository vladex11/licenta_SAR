"""
Modul pentru construirea vectorului de direcție (steering vector) pentru
tomografia SAR 4-D, conform formulării din:

    De Maio, A., Fornaro, G., Pauciullo, A. (2009).
    "Detection of Single Scatterers in Multidimensional SAR Imaging."
    IEEE Trans. Geosci. Remote Sens., vol. 47, no. 7, pp. 2284-2297.

Vectorul de direcție a(s, v) modelează contribuția unui dispersor cu
elevație reziduală s și viteză medie de deformare v la fazele observate
într-o stivă de N achiziții SAR.

Formula (3.2)-(3.3) din lucrarea de disertație:

    a_n(s, v) = (1/sqrt(N)) * exp[-j(4*pi/lambda)(b_n * s / r + t_n * v)]

unde:
    lambda    - lungimea de undă a sistemului SAR (m)
    b_n       - linia de bază perpendiculară a achiziției n (m)
    t_n       - distanța temporală a achiziției n față de master (ani)
    r         - distanța slant-range (m)
    s         - elevația reziduală a dispersorului (m)
    v         - viteza de deformare LOS (m/an, NOTĂ unitate!)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SystemGeometry:
    """
    Parametrii geometrici și fizici ai sistemului SAR și ai stivei.

    Atribute
    --------
    wavelength : float
        Lungimea de undă (m). Pentru Sentinel-1 banda C: 0.05546576 m.
    slant_range : float
        Distanța slant-range master-pixel (m). Pentru Sentinel-1: ~800 km.
    baselines_perp : np.ndarray, shape (N,)
        Liniile de bază perpendiculare ale celor N achiziții față de master (m).
        Master are baseline = 0.
    baselines_temp : np.ndarray, shape (N,)
        Distanțele temporale ale achizițiilor față de master, în ANI.
        Master are baseline = 0.
    """
    wavelength: float
    slant_range: float
    baselines_perp: np.ndarray
    baselines_temp: np.ndarray

    @property
    def n_images(self) -> int:
        return len(self.baselines_perp)

    @property
    def rayleigh_elevation(self) -> float:
        """Rezoluția Rayleigh în elevație (m): Δs = λ*r / (2*Bperp_span)"""
        bperp_span = self.baselines_perp.max() - self.baselines_perp.min()
        if bperp_span == 0:
            return np.inf
        return self.wavelength * self.slant_range / (2 * bperp_span)

    @property
    def rayleigh_velocity(self) -> float:
        """Rezoluția Rayleigh în viteză (m/an): Δv = λ / (2*T_total)"""
        t_span = self.baselines_temp.max() - self.baselines_temp.min()
        if t_span == 0:
            return np.inf
        return self.wavelength / (2 * t_span)

    def __post_init__(self):
        # Verificări de consistență
        self.baselines_perp = np.asarray(self.baselines_perp, dtype=np.float64)
        self.baselines_temp = np.asarray(self.baselines_temp, dtype=np.float64)
        if self.baselines_perp.shape != self.baselines_temp.shape:
            raise ValueError("baselines_perp și baselines_temp trebuie să aibă aceeași dimensiune")
        if self.baselines_perp.ndim != 1:
            raise ValueError("baseline-urile trebuie să fie vectori 1D")


def steering_vector(s: float, v: float, geom: SystemGeometry) -> np.ndarray:
    """
    Calculează vectorul de direcție a(s, v) ∈ ℂ^N pentru un dispersor
    cu elevație reziduală s (m) și viteză LOS v (m/an).

    Formula (3.2) din lucrare:
        a_n(s, v) = (1/sqrt(N)) * exp(-j * 4π/λ * (b_n * s / r + t_n * v))

    Parameters
    ----------
    s : float
        Elevația reziduală (m).
    v : float
        Viteza LOS (m/an). NOTĂ: pentru cm/an, înmulțește cu 0.01 înainte.
    geom : SystemGeometry
        Geometria sistemului și a stivei.

    Returns
    -------
    a : np.ndarray, shape (N,), dtype=complex128
        Vectorul de direcție normalizat (||a|| = 1).
    """
    N = geom.n_images
    phase = -(4.0 * np.pi / geom.wavelength) * (
        geom.baselines_perp * s / geom.slant_range + geom.baselines_temp * v
    )
    return np.exp(1j * phase) / np.sqrt(N)


def steering_dictionary(s_grid: np.ndarray, v_grid: np.ndarray,
                        geom: SystemGeometry) -> np.ndarray:
    """
    Calculează un dicționar de vectori de direcție pentru o grilă 2D
    de parametri (s, v).

    Returnează o matrice A de dimensiune (N, Ms * Mv), unde fiecare
    coloană este vectorul de direcție pentru un punct al grilei.

    Această reprezentare permite evaluarea eficientă a statisticilor de test
    prin operații matriciale (A.conj().T @ x).

    Parameters
    ----------
    s_grid : np.ndarray, shape (Ms,)
        Punctele grilei pentru elevația reziduală (m).
    v_grid : np.ndarray, shape (Mv,)
        Punctele grilei pentru viteza LOS (m/an).
    geom : SystemGeometry
        Geometria sistemului.

    Returns
    -------
    A : np.ndarray, shape (N, Ms * Mv), dtype=complex128
        Dicționarul de vectori de direcție.
        Coloana k = Ms*j + i corespunde punctului (s_grid[i], v_grid[j]).
    """
    N = geom.n_images
    Ms = len(s_grid)
    Mv = len(v_grid)

    # Construire eficientă cu broadcasting
    # phase[n, i, j] = -(4π/λ)(b_n * s_grid[i] / r + t_n * v_grid[j])
    bperp = geom.baselines_perp[:, None, None]  # (N, 1, 1)
    btemp = geom.baselines_temp[:, None, None]  # (N, 1, 1)
    s_g = s_grid[None, :, None]                 # (1, Ms, 1)
    v_g = v_grid[None, None, :]                 # (1, 1, Mv)

    phase = -(4.0 * np.pi / geom.wavelength) * (
        bperp * s_g / geom.slant_range + btemp * v_g
    )  # (N, Ms, Mv)

    A = np.exp(1j * phase) / np.sqrt(N)
    # Reshape la (N, Ms*Mv)
    return A.reshape(N, Ms * Mv)


def build_search_grid(geom: SystemGeometry,
                      s_range: tuple = None, v_range: tuple = None,
                      s_oversampling: int = 8, v_oversampling: int = 8,
                      min_points: int = 50):
    """
    Construiește grila de căutare 2D pentru (s, v) bazată pe rezoluțiile
    Rayleigh ale geometriei date.

    Parameters
    ----------
    geom : SystemGeometry
    s_range : tuple (s_min, s_max) în metri. Default: (-50, +50) m.
    v_range : tuple (v_min, v_max) în m/an. Default: (-0.03, +0.03) m/an = (-3, +3) cm/an.
    s_oversampling : int
        Factor de supra-eșantionare față de rezoluția Rayleigh (default 8).
    v_oversampling : int
    min_points : int
        Numărul minim de puncte ale grilei pe fiecare axă (default 50).

    Returns
    -------
    s_grid, v_grid : np.ndarray
        Vectorii de puncte ai grilei.
    """
    if s_range is None:
        s_range = (-50.0, 50.0)
    if v_range is None:
        v_range = (-0.03, 0.03)  # m/an = ±3 cm/an

    ds = geom.rayleigh_elevation / s_oversampling
    dv = geom.rayleigh_velocity / v_oversampling

    # Asigură numărul minim de puncte
    ds = min(ds, (s_range[1] - s_range[0]) / min_points)
    dv = min(dv, (v_range[1] - v_range[0]) / min_points)

    s_grid = np.arange(s_range[0], s_range[1] + ds / 2, ds)
    v_grid = np.arange(v_range[0], v_range[1] + dv / 2, dv)

    return s_grid, v_grid
