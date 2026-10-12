"""Spatial audio rendering for Physonic Phase 2A.

Public API:
    - HRTFSet: HRTF measurement container
    - BinauralMicrophone: Microphone with 3D position and orientation
    - load_kemar_hrtf: Load MIT KEMAR HRTF dataset
    - render_spatial_source: Render mono audio as binaural stereo
"""
from physonic.spatial.hrtf import HRTFSet, load_kemar_hrtf, interpolate_hrtf, convolve_hrtf
from physonic.spatial.microphone import BinauralMicrophone
from physonic.spatial.renderer import render_spatial_source

__all__ = [
    "HRTFSet",
    "BinauralMicrophone",
    "load_kemar_hrtf",
    "interpolate_hrtf",
    "convolve_hrtf",
    "render_spatial_source",
]
