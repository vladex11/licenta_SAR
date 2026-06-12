# GLRT-SAR: Detecția dispersorilor stabili în stive multi-temporale SAR

**Lucrare de disertație** — Panait Vlad-Marian  
**Coordonator:** Ș.l. dr. ing. Cosmin Dănișor  
**Facultatea de Electronică, Telecomunicații și Tehnologia Informației — UPB**

## Descrierea proiectului

Implementare Python a detectoarelor **SL-GLRT** (Single-Look GLRT) și **ML-GLRT** (Multi-Look GLRT) pentru identificarea dispersorilor stabili (Persistent Scatterers) în stive multi-temporale de imagini SAR, conform formulărilor din:

- De Maio, Fornaro, Pauciullo (2009) — SL-GLRT
- Pauciullo, Reale, Franzé, Fornaro (2018) — ML-GLRT
- Dănișor, Pauciullo, Reale, Fornaro (2023) — analiză comparativă

Zona de studiu: **Curtea de Argeș, România** (date Sentinel-1).

## Structura proiectului

```
glrt_sar/
├── detectors/              # Implementări detectoare
│   ├── sl_glrt.py         # SL-GLRT (single-look)
│   ├── ml_glrt.py         # ML-GLRT (multi-look)
│   └── steering.py        # Vectorul de direcție a(s,v)
│
├── simulation/             # Validare Monte Carlo
│   ├── generate_data.py   # Generare date sintetice
│   ├── monte_carlo.py     # Simulări Pfa și Pd
│   └── threshold.py       # Calcul praguri T(Pfa, N, L)
│
├── real_data/              # Procesare date reale
│   ├── load_snap.py       # Citire stive din SNAP (BEAM-DIMAP/GeoTIFF)
│   ├── baselines.py       # Calcul/citire baseline-uri
│   └── process_stack.py   # Pipeline complet pe imagini reale
│
├── utils/                  # Utilități
│   ├── plotting.py        # Funcții de vizualizare
│   └── metrics.py         # Metrici (ENL, contrast, coerență)
│
├── config/                 # Configurări
│   └── sentinel1.py       # Parametri sistem Sentinel-1
│
├── notebooks/              # Jupyter Notebooks demo
│   ├── 01_simulare_validare.ipynb
│   ├── 02_curbe_Pfa_Pd.ipynb
│   ├── 03_comparatie_SL_ML.ipynb
│   └── 04_date_reale_CurteaArges.ipynb
│
├── tests/                  # Teste unitare
│   └── test_detectors.py
│
└── requirements.txt        # Dependențe Python
```

## Instalare

```bash
# Creează un mediu virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# sau: venv\Scripts\activate  # Windows

# Instalează dependențele
pip install -r requirements.txt
```

## Utilizare rapidă

```python
from glrt_sar.detectors import sl_glrt, ml_glrt
from glrt_sar.simulation import generate_data

# Generează date simulate
x, ground_truth = generate_data.simulate_pixel(N=30, snr_db=10, s_true=20, v_true=0.5)

# Aplică SL-GLRT
detector = sl_glrt.SLGLRT(baselines_perp=baselines_perp, baselines_temp=baselines_temp,
                          slant_range=800e3, wavelength=0.0555)
result = detector.detect(x, threshold=0.5)
print(f"Detectat: {result.detected}, Elevație: {result.s_est:.2f} m, Viteză: {result.v_est:.2f} cm/an")
```

## Workflow tipic

1. **Validare pe date simulate** (notebook 01-03):
   - Simulare Monte Carlo cu N realizări sub H₀ → curbe Pfa vs T
   - Simulare cu dispersor injectat → curbe Pd vs SNR
   - Comparație SL vs ML pentru diverse N și L

2. **Procesare date reale** (notebook 04):
   - Citire stivă pre-procesată în SNAP (Curtea de Argeș)
   - Aplicare SL-GLRT și ML-GLRT
   - Generare hărți de detecție și estimări (s, v)
   - Vizualizare pe imagine de bază
