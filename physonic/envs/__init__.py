"""Gymnasium RL environments for acoustic perception tasks."""

from physonic.envs.localization import SoundLocalizationEnv
from physonic.envs.classification import SoundClassificationEnv
from physonic.envs.selective_response import SelectiveResponseEnv
from physonic.envs.base import BaseAcousticEnv
from physonic.envs.feature_extractors import (
    AudioCNNExtractor,
    AudioAttentionExtractor,
    FrameStackExtractor,
    get_audio_policy_kwargs,
)

__all__ = [
    "SoundLocalizationEnv",
    "SoundClassificationEnv",
    "SelectiveResponseEnv",
    "BaseAcousticEnv",
    # Feature extractors for SB3
    "AudioCNNExtractor",
    "AudioAttentionExtractor",
    "FrameStackExtractor",
    "get_audio_policy_kwargs",
]
