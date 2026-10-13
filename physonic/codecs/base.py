"""Base codec engine interface."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseCodecEngine(ABC):
    """Abstract base class for audio codec engines."""

    @abstractmethod
    def encode(self, audio_data: np.ndarray, sample_rate: int) -> dict[str, Any]:
        """Encode audio to compressed representation.

        Args:
            audio_data: Audio array, shape (channels, samples) or (samples,)
            sample_rate: Sample rate in Hz

        Returns:
            Dictionary containing encoded data and metadata
        """
        pass

    @abstractmethod
    def decode(self, encoded_data: dict[str, Any]) -> tuple[np.ndarray, int]:
        """Decode compressed representation back to audio.

        Args:
            encoded_data: Dictionary from encode() method

        Returns:
            Tuple of (audio_array, sample_rate)
            Audio array has shape (channels, samples)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return codec name."""
        pass
