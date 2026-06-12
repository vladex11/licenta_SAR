"""
Modul pentru citirea stivelor de imagini SAR pre-procesate în SNAP.

Workflow tipic în SNAP pentru a obține datele pe care le citim:

  1. Read SLC products (Sentinel-1 IW SLC)
  2. TOPSAR-Split (selectare subswath și burst-uri)
  3. Apply Orbit File (orbite precise)
  4. Back-Geocoding (co-registrare la imaginea master)
  5. Enhanced Spectral Diversity (rafinare co-registrare)
  6. Interferogram formation (cu Flat-Earth phase removal)
  7. TOPSAR-Deburst
  8. Multilook + Subset (opțional, pentru zona de interes)
  9. Goldstein Phase Filtering (opțional)
  10. Export: GeoTIFF complex sau BEAM-DIMAP

Pentru detecția GLRT avem nevoie de:
  - Stivă SLC co-registrată (valori complexe)
  - Liniile de bază perpendiculare (din SNAP > InSAR Stack Overview)
  - Liniile de bază temporale (datele de achiziție)
  - Distanța slant-range (din metadate SNAP)

Acest modul nu apelează SNAP direct (utilizatorul face procesarea
manual), ci doar citește produsele exportate.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SARStack:
    """
    Reprezentarea unei stive SAR co-registrate.

    Atribute
    --------
    data : np.ndarray, shape (H, W, N), complex
        Stiva de date complexe (după co-registrare în SNAP).
    dates : list of datetime
        Datele de achiziție ale celor N imagini.
    baselines_perp : np.ndarray, shape (N,)
        Linii de bază perpendiculare (m).
    master_idx : int
        Indexul imaginii master.
    slant_range : float
        Distanța slant-range (m).
    incidence_angle : float
        Unghiul de incidență (grade), opțional.
    """
    data: np.ndarray
    dates: list
    baselines_perp: np.ndarray
    master_idx: int = 0
    slant_range: float = 800e3
    incidence_angle: float = 39.0

    @property
    def n_images(self):
        return self.data.shape[2]

    @property
    def shape_spatial(self):
        return self.data.shape[:2]

    def baselines_temporal_years(self) -> np.ndarray:
        """Returnează vectorul de baseline-uri temporale (în ani) față de master."""
        master_date = self.dates[self.master_idx]
        days = np.array([(d - master_date).days for d in self.dates])
        return days / 365.25

    def to_geometry(self):
        """Convertește la SystemGeometry pentru utilizare cu detectoarele."""
        from ..detectors.steering import SystemGeometry
        from ..config.sentinel1 import WAVELENGTH_C

        return SystemGeometry(
            wavelength=WAVELENGTH_C,
            slant_range=self.slant_range,
            baselines_perp=self.baselines_perp,
            baselines_temp=self.baselines_temporal_years(),
        )


def load_geotiff_stack(folder: str, pattern: str = "*.tif",
                       dates_file: Optional[str] = None,
                       baselines_file: Optional[str] = None,
                       master_idx: int = 0) -> SARStack:
    """
    Citește o stivă de imagini GeoTIFF complex exportate din SNAP.

    Așteaptă structura:
        folder/
            img_001.tif  (SLC complex, exportat din SNAP)
            img_002.tif
            ...
            dates.txt           # opțional - lista datelor
            baselines.txt       # opțional - lista baseline-urilor perp.

    Format dates.txt: o dată pe linie, format YYYYMMDD sau YYYY-MM-DD
    Format baselines.txt: o valoare pe linie (m), în aceeași ordine ca imaginile.

    Parameters
    ----------
    folder : str
    pattern : str
    dates_file : str sau None
    baselines_file : str sau None
    master_idx : int

    Returns
    -------
    SARStack
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("Pachetul 'rasterio' este necesar. Instalează cu: pip install rasterio")

    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nu s-au găsit imagini în {folder} cu pattern {pattern}")

    print(f"Citire {len(files)} imagini din {folder}...")

    # Citire prima imagine pentru a determina dimensiunea
    with rasterio.open(files[0]) as src:
        H, W = src.height, src.width
        dtype = np.complex64

    # Alocă stiva
    data = np.zeros((H, W, len(files)), dtype=dtype)

    for i, f in enumerate(files):
        with rasterio.open(f) as src:
            # Pentru SLC, GeoTIFF poate avea banda complexă direct sau 2 benzi (real, imag)
            if src.count == 1:
                data[..., i] = src.read(1)
            elif src.count == 2:
                # banda 1 = real, banda 2 = imaginar
                data[..., i] = src.read(1).astype(np.float32) + 1j * src.read(2).astype(np.float32)
            else:
                raise ValueError(f"Format neașteptat pentru {f}: {src.count} benzi")

    # Citire date
    if dates_file:
        import datetime
        with open(folder / dates_file) as f:
            dates = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Acceptă YYYYMMDD sau YYYY-MM-DD
                line = line.replace('-', '').replace('/', '')
                dates.append(datetime.datetime.strptime(line[:8], '%Y%m%d').date())
    else:
        # Generează date sintetice: una la fiecare 6 zile, începând cu master
        import datetime
        base = datetime.date(2023, 1, 1)
        dates = [base + datetime.timedelta(days=6 * i) for i in range(len(files))]
        print("AVERTISMENT: dates.txt lipsă. S-au generat date sintetice la 6 zile.")

    # Citire baseline-uri
    if baselines_file:
        baselines = np.loadtxt(folder / baselines_file)
    else:
        # Generează baseline-uri sintetice (uniform în [-150, 150] m)
        rng = np.random.default_rng(42)
        baselines = rng.uniform(-150, 150, len(files))
        baselines[master_idx] = 0.0
        print("AVERTISMENT: baselines.txt lipsă. S-au generat baseline-uri sintetice.")

    return SARStack(
        data=data,
        dates=dates,
        baselines_perp=np.asarray(baselines),
        master_idx=master_idx,
    )

def load_snap_multiband_geotiff(filepath, dates_file=None, baselines_file=None, master_idx=0):
    import datetime
    import rasterio
    import numpy as np
    from pathlib import Path

    filepath = Path(filepath)

    with rasterio.open(filepath) as src:
        H, W = src.height, src.width
        count = src.count

        if count % 3 != 0:
            raise ValueError(f"Expected 3 bands per acquisition, got {count} bands")

        N = count // 3
        data = np.zeros((H, W, N), dtype=np.complex64)

        for k in range(N):
            real_band = 3 * k + 1
            imag_band = 3 * k + 2

            real = src.read(real_band).astype(np.float32)
            imag = src.read(imag_band).astype(np.float32)

            data[:, :, k] = real + 1j * imag

    folder = filepath.parent

    if dates_file:
        dates = []
        with open(folder / dates_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    line = line.replace("-", "").replace("/", "")
                    dates.append(datetime.datetime.strptime(line[:8], "%Y%m%d").date())
    else:
        date_strings = [
            "20260314",
            "20260125",
            "20260206",
            "20260218",
            "20260302",
            "20260326",
            "20260407",
            "20260419",
            "20260501",
            "20260513",
        ]
        dates = [datetime.datetime.strptime(d, "%Y%m%d").date() for d in date_strings]

    if baselines_file:
        baselines = np.loadtxt(folder / baselines_file)
    else:
        baselines = np.zeros(N, dtype=float)

    return SARStack(
        data=data,
        dates=dates,
        baselines_perp=np.asarray(baselines),
        master_idx=master_idx,
    )

def load_npz_stack(filepath: str) -> SARStack:
    """
    Citește o stivă din format .npz (format Python nativ, util după ce
    ai convertit datele din SNAP prin script).

    Așteaptă un fișier .npz cu cheile:
        data           - (H, W, N) complex64
        dates          - (N,) string YYYYMMDD
        baselines_perp - (N,) float
        master_idx     - scalar int
        slant_range    - scalar float (opțional)
    """
    import datetime
    arr = np.load(filepath, allow_pickle=False)

    dates_raw = arr['dates']
    dates = [datetime.datetime.strptime(str(d), '%Y%m%d').date() for d in dates_raw]

    return SARStack(
        data=arr['data'],
        dates=dates,
        baselines_perp=arr['baselines_perp'],
        master_idx=int(arr.get('master_idx', 0)),
        slant_range=float(arr.get('slant_range', 800e3)),
        incidence_angle=float(arr.get('incidence_angle', 39.0)),
    )


def save_npz_stack(stack: SARStack, filepath: str):
    """Salvează SARStack în format .npz pentru utilizare ulterioară."""
    dates_str = np.array([d.strftime('%Y%m%d') for d in stack.dates])
    np.savez_compressed(
        filepath,
        data=stack.data.astype(np.complex64),
        dates=dates_str,
        baselines_perp=stack.baselines_perp,
        master_idx=stack.master_idx,
        slant_range=stack.slant_range,
        incidence_angle=stack.incidence_angle,
    )
    print(f"Stiva salvată în {filepath} "
          f"({stack.n_images} imagini, {stack.shape_spatial[0]}×{stack.shape_spatial[1]})")
