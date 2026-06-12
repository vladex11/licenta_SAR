"""
Configurări specifice constelației Sentinel-1 (banda C).

Folosit ca referință pentru:
- Generarea datelor simulate cu parametri realistici
- Procesarea datelor reale Sentinel-1 SLC IW

Sursa parametrilor: ESA Sentinel-1 User Handbook (2013) și documentația
oficială Sentinel-1.
"""

# === Parametri sistem Sentinel-1 ===
WAVELENGTH_C = 0.05546576  # m (lungimea de undă banda C, ~5.55 cm)
FREQUENCY_C = 5.405e9      # Hz (frecvența nominală)

# === Distanțe slant-range tipice (IW mode) ===
SLANT_RANGE_NEAR = 730e3   # m (near range pentru IW1)
SLANT_RANGE_FAR = 870e3    # m (far range pentru IW3)
SLANT_RANGE_TYPICAL = 800e3  # m (valoare medie utilă pentru simulări)

# === Unghiuri de incidență (IW mode) ===
INCIDENCE_NEAR = 29.1  # grade
INCIDENCE_FAR = 46.0   # grade
INCIDENCE_TYPICAL = 39.0  # grade

# === Revizitare temporală ===
REVISIT_DAYS_SINGLE_SAT = 12  # un singur satelit (S-1A sau S-1B)
REVISIT_DAYS_CONSTELLATION = 6  # constelația S-1A + S-1B (operațional 2016-2021)

# === Rezoluție în mode IW (interferometric) ===
RESOLUTION_AZIMUTH = 20.0  # m
RESOLUTION_RANGE = 5.0     # m (slant range)
RESOLUTION_GROUND_RANGE = 22.0  # m

# === Orbita ===
ORBIT_ALTITUDE = 693e3     # m
ORBIT_INCLINATION = 98.18  # grade (Sun-synchronous)


def create_sentinel1_geometry(baselines_perp, baselines_temp_days,
                              slant_range: float = SLANT_RANGE_TYPICAL):
    """
    Wrapper pentru a crea rapid o SystemGeometry pentru Sentinel-1.

    Parameters
    ----------
    baselines_perp : array_like, m
    baselines_temp_days : array_like, zile (se convertește la ani)
    slant_range : float, m

    Returns
    -------
    SystemGeometry pre-completată cu parametrii Sentinel-1.
    """
    from ..detectors.steering import SystemGeometry
    import numpy as np

    baselines_temp_years = np.asarray(baselines_temp_days) / 365.25

    return SystemGeometry(
        wavelength=WAVELENGTH_C,
        slant_range=slant_range,
        baselines_perp=np.asarray(baselines_perp),
        baselines_temp=baselines_temp_years,
    )


# === Curtea de Argeș - parametri zonă de studiu ===
CURTEA_DE_ARGES = {
    'lat': 45.135,
    'lon': 24.673,
    'description': (
        'Oraș istoric din județul Argeș, România. '
        'Conține Mănăstirea Curtea de Argeș (sec. XVI) și Biserica '
        'Domnească (sec. XIV) - structuri patrimoniale relevante '
        'pentru monitorizarea deformărilor. Zona include și versanți '
        'instabili din Sub-Carpații Getici.'
    ),
    # Acoperire Sentinel-1
    'sentinel1_tracks': {
        'ascending': 7,    # track 7 (orbit cycle)
        'descending': 80,  # track 80
    },
    # Bounding box recomandat pentru SLC subset (zoom pe oraș + împrejurimi)
    'aoi_bbox': {
        'lat_min': 45.10, 'lat_max': 45.17,
        'lon_min': 24.62, 'lon_max': 24.73,
    },
}
