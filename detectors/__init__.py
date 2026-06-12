"""Modulul detectors expune SL-GLRT și ML-GLRT."""
from .steering import SystemGeometry, steering_vector, steering_dictionary, build_search_grid
from .sl_glrt import SLGLRT, DetectionResult
from .ml_glrt import MLGLRT

__all__ = [
    'SystemGeometry', 'steering_vector', 'steering_dictionary', 'build_search_grid',
    'SLGLRT', 'MLGLRT', 'DetectionResult',
]
