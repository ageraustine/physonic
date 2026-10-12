"""Base environment class for acoustic perception tasks."""

import gymnasium as gym
from abc import ABC, abstractmethod


class BaseAcousticEnv(gym.Env, ABC):
    """
    Base class for acoustic perception environments.

    Provides shared utilities:
    - HRTF loading and caching
    - Audio generation from physics
    - Microphone configuration
    - Rendering utilities
    """

    def __init__(self, hrtf_path: str = "assets/hrtf/mit_kemar/"):
        super().__init__()
        self.hrtf_path = hrtf_path
        self._hrtf = None

    @property
    def hrtf(self):
        """Lazy-load HRTF (expensive operation)."""
        if self._hrtf is None:
            from physonic.spatial import load_kemar_hrtf
            self._hrtf = load_kemar_hrtf(self.hrtf_path)
        return self._hrtf

    @abstractmethod
    def _generate_audio(self):
        """Generate audio for current episode (implemented by subclass)."""
        pass
