"""Visualization and demo tools for Physonic environments.

This module provides tools for:
- Recording RL episodes (Gymnasium compatible)
- Visualizing audio observations (waveforms, spectrograms)
- Interactive environment viewing
- Exporting to HuggingFace and Isaac Sim formats
"""

from physonic.visualization.recorder import EpisodeRecorder
from physonic.visualization.plots import (
    plot_waveform,
    plot_spectrogram,
    plot_mel_spectrogram,
    plot_metrics,
)

__all__ = [
    "EpisodeRecorder",
    "plot_waveform",
    "plot_spectrogram",
    "plot_mel_spectrogram",
    "plot_metrics",
]
