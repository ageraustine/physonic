"""Sound synthesis module for non-impact sounds.

This module provides synthesis methods for:
- Continuous equipment sounds (fans, servers, HVAC)
- Alert/beep sounds (alarms, warnings)
- Sound mixing (impact + continuous + alert)
"""

from physonic.synthesis.continuous import (
    ContinuousSoundSpec,
    synthesize_continuous_sound,
    create_preset_sound,
    synthesize_multiple_sources,
)
from physonic.synthesis.alerts import (
    AlertSoundSpec,
    synthesize_alert_sound,
    create_preset_alert,
    synthesize_two_tone_alert,
    place_alert_in_timeline,
)
from physonic.synthesis.mixer import (
    mix_acoustic_scene,
    mix_with_layers,
    create_data_center_ambient,
)

__all__ = [
    # Continuous sounds
    "ContinuousSoundSpec",
    "synthesize_continuous_sound",
    "create_preset_sound",
    "synthesize_multiple_sources",
    # Alert sounds
    "AlertSoundSpec",
    "synthesize_alert_sound",
    "create_preset_alert",
    "synthesize_two_tone_alert",
    "place_alert_in_timeline",
    # Mixing
    "mix_acoustic_scene",
    "mix_with_layers",
    "create_data_center_ambient",
]
