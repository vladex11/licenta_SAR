"""
GLRT-SAR: Detecția dispersorilor stabili în stive multi-temporale SAR.

Pachet Python pentru lucrarea de disertație:
"Estimarea parametrilor scenelor folosind seturi multi-temporale
de imagini SAR. Detecția țintelor stabile pe baza raportului
de plauzibilitate generalizat."

Autor: Panait Vlad-Marian
Coordonator: Ș.l. dr. ing. Cosmin Dănișor
UPB - ETTI, 2026
"""

__version__ = "0.1.0"
__author__ = "Panait Vlad-Marian"

from . import detectors
from . import simulation
from . import utils
from . import config
from . import real_data
