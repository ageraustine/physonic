"""Neural audio codecs for efficient audio compression.

This module provides neural codec engines for compressing audio observations
in RL environments, reducing memory footprint and enabling efficient training
with high-fidelity audio.

Available codecs:
- Mimi: Kyutai's 1.1kbps neural codec (12.5Hz frame rate)
"""

from physonic.codecs.engines.mimi import MimiAudioEngine

__all__ = [
    "MimiAudioEngine",
]
