"""Custom feature extractors for acoustic RL environments.

Handles multi-modal observations:
- Audio codes: 2D temporal data (32 codebooks × T frames) → CNN
- Ultrasonic: 1D vector → MLP
- State: 1D vector → MLP

These extractors properly preserve temporal structure in audio
instead of naive flattening.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict


class AudioCNNExtractor(BaseFeaturesExtractor):
    """Feature extractor with CNN for audio codes.

    Architecture:
    - Audio codes (32, T): 2D CNN → features
    - Ultrasonic (4,): Linear → features
    - State (6,): Linear → features
    - Concatenate all → output

    This preserves temporal structure in audio instead of flattening.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        audio_features: int = 128,
        ultrasonic_features: int = 32,
        state_features: int = 32,
    ):
        # Calculate total features
        features_dim = audio_features + ultrasonic_features + state_features
        super().__init__(observation_space, features_dim=features_dim)

        # Get observation shapes
        audio_shape = observation_space["audio_codes"].shape  # (32, T)
        ultrasonic_dim = observation_space["ultrasonic"].shape[0]  # 4
        state_dim = observation_space["state"].shape[0]  # 6

        n_codebooks, n_frames = audio_shape

        # Audio CNN: treats (32, T) as single-channel 2D image
        # Conv2d expects (batch, channels, height, width)
        self.audio_cnn = nn.Sequential(
            # Input: (batch, 1, 32, T)
            nn.Conv2d(1, 32, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, 1, n_codebooks, n_frames)
            cnn_out_size = self.audio_cnn(sample).shape[1]

        # Audio MLP: CNN output → audio_features
        self.audio_mlp = nn.Sequential(
            nn.Linear(cnn_out_size, audio_features),
            nn.ReLU(),
        )

        # Ultrasonic MLP
        self.ultrasonic_mlp = nn.Sequential(
            nn.Linear(ultrasonic_dim, ultrasonic_features),
            nn.ReLU(),
        )

        # State MLP
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, state_features),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Audio: (batch, 32, T) → (batch, 1, 32, T) for Conv2d
        audio = observations["audio_codes"].float()
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # Add batch dim if missing
        audio = audio.unsqueeze(1)  # Add channel dim: (batch, 1, 32, T)

        # Normalize audio codes to [0, 1] range (codebook indices 0-2047)
        audio = audio / 2047.0

        audio_features = self.audio_mlp(self.audio_cnn(audio))

        # Ultrasonic
        ultrasonic = observations["ultrasonic"].float()
        if ultrasonic.dim() == 1:
            ultrasonic = ultrasonic.unsqueeze(0)
        ultrasonic_features = self.ultrasonic_mlp(ultrasonic)

        # State
        state = observations["state"].float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state_features = self.state_mlp(state)

        # Concatenate all features
        return torch.cat([audio_features, ultrasonic_features, state_features], dim=1)


class AudioAttentionExtractor(BaseFeaturesExtractor):
    """Feature extractor with self-attention for audio codes.

    Better for capturing long-range temporal dependencies in audio.
    Uses transformer-style attention over time frames.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        audio_features: int = 128,
        ultrasonic_features: int = 32,
        state_features: int = 32,
        n_heads: int = 4,
    ):
        features_dim = audio_features + ultrasonic_features + state_features
        super().__init__(observation_space, features_dim=features_dim)

        # Get observation shapes
        audio_shape = observation_space["audio_codes"].shape  # (32, T)
        ultrasonic_dim = observation_space["ultrasonic"].shape[0]
        state_dim = observation_space["state"].shape[0]

        n_codebooks, _ = audio_shape

        # Audio: embed codebooks, then attention over time
        self.codebook_embed = nn.Linear(n_codebooks, 64)

        # Self-attention over time frames
        self.audio_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=n_heads,
            batch_first=True,
        )

        # Pool over time and project
        self.audio_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, audio_features),
            nn.ReLU(),
        )

        # Ultrasonic MLP
        self.ultrasonic_mlp = nn.Sequential(
            nn.Linear(ultrasonic_dim, ultrasonic_features),
            nn.ReLU(),
        )

        # State MLP
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, state_features),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Audio: (batch, 32, T) → embed → attention
        audio = observations["audio_codes"].float()
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        # Normalize and transpose: (batch, 32, T) → (batch, T, 32)
        audio = (audio / 2047.0).transpose(1, 2)

        # Embed codebooks: (batch, T, 32) → (batch, T, 64)
        audio = self.codebook_embed(audio)

        # Self-attention: (batch, T, 64) → (batch, T, 64)
        audio, _ = self.audio_attention(audio, audio, audio)

        # Pool over time: (batch, T, 64) → (batch, 64) → (batch, audio_features)
        audio = audio.transpose(1, 2)  # (batch, 64, T)
        audio_features = self.audio_pool(audio)

        # Ultrasonic
        ultrasonic = observations["ultrasonic"].float()
        if ultrasonic.dim() == 1:
            ultrasonic = ultrasonic.unsqueeze(0)
        ultrasonic_features = self.ultrasonic_mlp(ultrasonic)

        # State
        state = observations["state"].float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state_features = self.state_mlp(state)

        return torch.cat([audio_features, ultrasonic_features, state_features], dim=1)


class FrameStackExtractor(BaseFeaturesExtractor):
    """Simple extractor that works with frame-stacked observations.

    When using VecFrameStack, observations are stacked along a new dimension.
    This extractor handles the stacked format properly.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        # Calculate flattened input size
        total_size = 0
        for key, space in observation_space.spaces.items():
            total_size += np.prod(space.shape)

        # Simple MLP (frame stacking already provides temporal info)
        self.mlp = nn.Sequential(
            nn.Linear(total_size, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Flatten all observations (sorted keys for deterministic order)
        flat_obs = []
        for obs_key in sorted(observations.keys()):
            obs = observations[obs_key].float()
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            flat_obs.append(obs.flatten(start_dim=1))

        combined = torch.cat(flat_obs, dim=1)
        return self.mlp(combined)


# Convenience function to get policy kwargs
def get_audio_policy_kwargs(
    extractor_type: str = "cnn",
    audio_features: int = 128,
    net_arch: list = None,
) -> dict:
    """Get policy kwargs for audio-based navigation.

    Args:
        extractor_type: "cnn", "attention", or "mlp"
        audio_features: Number of audio feature dimensions
        net_arch: Network architecture for actor/critic

    Returns:
        policy_kwargs dict for SB3
    """
    if net_arch is None:
        net_arch = dict(pi=[256, 256], vf=[256, 256])

    if extractor_type == "cnn":
        extractor_class = AudioCNNExtractor
    elif extractor_type == "attention":
        extractor_class = AudioAttentionExtractor
    else:
        extractor_class = FrameStackExtractor

    return dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=dict(
            audio_features=audio_features,
        ) if extractor_type != "mlp" else dict(),
        net_arch=net_arch,
    )
