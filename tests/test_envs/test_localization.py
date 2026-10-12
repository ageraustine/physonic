"""Tests for sound localization environment."""

import pytest
import numpy as np

from physonic.envs import SoundLocalizationEnv

# Use test HRTF dataset for all tests
TEST_HRTF_PATH = "assets/hrtf/test_kemar/"


class TestSoundLocalizationEnv:
    """Test SoundLocalizationEnv basic functionality."""

    def test_env_creation(self):
        """Test environment can be created."""
        env = SoundLocalizationEnv(use_mel_spectrogram=True, hrtf_path=TEST_HRTF_PATH)
        assert env.observation_space.shape[0] == 64  # n_mels
        assert env.action_space.shape == (1,)

    def test_reset_returns_valid_observation(self):
        """Test reset() returns valid observation."""
        env = SoundLocalizationEnv(hrtf_path=TEST_HRTF_PATH)
        obs, info = env.reset()

        # Check observation shape
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

        # Check info contains required keys
        assert "true_azimuth" in info
        assert "true_elevation" in info
        assert "source_pos" in info
        assert "distance" in info

        # Check azimuth is in valid range
        assert -180.0 <= info["true_azimuth"] <= 180.0

    def test_step_computes_reward(self):
        """Test step() computes angular error reward."""
        env = SoundLocalizationEnv(hrtf_path=TEST_HRTF_PATH)
        obs, info = env.reset()
        true_az = info["true_azimuth"]

        # Perfect prediction (reward = 0.0)
        obs, reward, done, trunc, info = env.step([true_az])
        assert reward == pytest.approx(0.0)
        assert done is True
        assert trunc is False

        # 90° error (reward ≈ -0.5)
        obs, info = env.reset()  # Get new info from reset
        obs, reward, done, trunc, info = env.step([info["true_azimuth"] + 90])
        assert -0.6 < reward < -0.4

    def test_observation_space_matches(self):
        """Test obs shape matches declared space."""
        env = SoundLocalizationEnv(use_mel_spectrogram=True, hrtf_path=TEST_HRTF_PATH)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape

    def test_raw_audio_mode(self):
        """Test raw audio observation mode."""
        env = SoundLocalizationEnv(use_mel_spectrogram=False, audio_duration=1.0, hrtf_path=TEST_HRTF_PATH)
        obs, _ = env.reset()
        assert obs.shape == (44100, 2)  # 1s stereo @ 44.1kHz
        assert obs.dtype == np.float32

    def test_action_space_bounds(self):
        """Test action space has correct bounds."""
        env = SoundLocalizationEnv(hrtf_path=TEST_HRTF_PATH)
        assert env.action_space.low[0] == -180.0
        assert env.action_space.high[0] == 180.0

    def test_reward_range(self):
        """Test reward is in [-1, 0] range."""
        env = SoundLocalizationEnv(hrtf_path=TEST_HRTF_PATH)
        obs, info = env.reset()

        # Test various predictions
        for azimuth_offset in [0, 45, 90, 135, 180]:
            pred = info["true_azimuth"] + azimuth_offset
            obs, reward, done, trunc, info = env.step([pred])
            assert -1.0 <= reward <= 0.0


class TestMultipleEpisodes:
    """Test multiple reset/step cycles."""

    def test_multiple_episodes(self):
        """Test environment can be reset multiple times."""
        env = SoundLocalizationEnv(hrtf_path=TEST_HRTF_PATH)

        for _ in range(10):
            obs, info = env.reset()
            assert obs.shape == env.observation_space.shape

            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            assert done

    def test_different_azimuths(self):
        """Test different episodes have different source positions."""
        env = SoundLocalizationEnv(hrtf_path=TEST_HRTF_PATH)

        azimuths = []
        for _ in range(20):
            obs, info = env.reset()
            azimuths.append(info["true_azimuth"])

        # Should have variety (not all the same)
        assert len(set(azimuths)) > 10


class TestAudioGeneration:
    """Test audio generation variations."""

    def test_different_materials(self):
        """Test audio generation with different materials."""
        for material in ["steel", "glass", "wood", "rubber"]:
            env = SoundLocalizationEnv(material=material, hrtf_path=TEST_HRTF_PATH)
            obs, info = env.reset()
            assert obs.shape == env.observation_space.shape

    def test_different_scenarios(self):
        """Test different scenario types."""
        for scenario in ["drop", "slide"]:
            env = SoundLocalizationEnv(scenario_type=scenario, hrtf_path=TEST_HRTF_PATH)
            obs, info = env.reset()
            assert obs.shape == env.observation_space.shape

    def test_audio_duration(self):
        """Test different audio durations."""
        for duration in [0.5, 1.0, 2.0]:
            env = SoundLocalizationEnv(
                audio_duration=duration,
                use_mel_spectrogram=False,
                hrtf_path=TEST_HRTF_PATH
            )
            obs, _ = env.reset()
            expected_samples = int(duration * 44100)
            assert obs.shape[0] == expected_samples


class TestSeedReproducibility:
    """Test seed reproducibility."""

    def test_reset_with_seed(self):
        """Test reset with seed produces same results."""
        env = SoundLocalizationEnv(hrtf_path=TEST_HRTF_PATH)

        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)

        assert info1["true_azimuth"] == pytest.approx(info2["true_azimuth"])
        assert np.allclose(obs1, obs2)

    def test_different_seeds(self):
        """Test different seeds produce different results."""
        env = SoundLocalizationEnv(hrtf_path=TEST_HRTF_PATH)

        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=123)

        # Should be different (very unlikely to be same)
        assert info1["true_azimuth"] != pytest.approx(info2["true_azimuth"])


@pytest.mark.integration
class TestGymnasiumCompliance:
    """Test Gymnasium API compliance."""

    def test_gymnasium_check_env(self):
        """Test environment passes Gymnasium checks."""
        from gymnasium.utils.env_checker import check_env

        env = SoundLocalizationEnv(audio_duration=0.5, hrtf_path=TEST_HRTF_PATH)  # Shorter for faster test
        check_env(env.unwrapped, skip_render_check=True)

    def test_reset_returns_tuple(self):
        """Test reset returns (obs, info) tuple."""
        env = SoundLocalizationEnv(hrtf_path=TEST_HRTF_PATH)
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_step_returns_five_tuple(self):
        """Test step returns (obs, reward, terminated, truncated, info)."""
        env = SoundLocalizationEnv(hrtf_path=TEST_HRTF_PATH)
        env.reset()
        result = env.step(env.action_space.sample())
        assert isinstance(result, tuple)
        assert len(result) == 5


@pytest.mark.integration
class TestInfoDict:
    """Test info dictionary contents."""

    def test_info_contains_ground_truth(self):
        """Test info dict contains ground truth labels."""
        env = SoundLocalizationEnv(hrtf_path=TEST_HRTF_PATH)
        obs, info = env.reset()

        required_keys = ["true_azimuth", "true_elevation", "source_pos", "distance"]
        for key in required_keys:
            assert key in info

    def test_step_info_contains_prediction(self):
        """Test step info contains prediction details."""
        env = SoundLocalizationEnv(hrtf_path=TEST_HRTF_PATH)
        obs, info = env.reset()
        obs, reward, done, trunc, info = env.step([45.0])

        assert "predicted_azimuth" in info
        assert "angular_error" in info
        assert info["predicted_azimuth"] == pytest.approx(45.0)
