"""Tests for sound classification environment."""

import pytest
import numpy as np

from physonic.envs import SoundClassificationEnv

# Use test HRTF dataset for all tests
TEST_HRTF_PATH = "assets/hrtf/test_kemar/"


class TestSoundClassificationEnv:
    """Test SoundClassificationEnv basic functionality."""

    def test_env_creation(self):
        """Test environment can be created."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)
        assert env.action_space.n == 4  # 4 material classes
        assert env.observation_space.shape[0] == 64  # n_mels

    def test_reset_returns_valid_observation(self):
        """Test reset() returns valid observation."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)
        obs, info = env.reset()

        assert obs.shape == env.observation_space.shape
        assert "true_material" in info
        assert "true_material_idx" in info
        assert info["true_material"] in env.MATERIAL_CLASSES
        assert 0 <= info["true_material_idx"] < 4

    def test_step_computes_reward(self):
        """Test step() computes classification reward."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)
        obs, info = env.reset()
        true_idx = info["true_material_idx"]

        # Correct prediction (reward = 1.0)
        obs, reward, done, trunc, info = env.step(true_idx)
        assert reward == 1.0
        assert info["correct"] is True
        assert done is True

        # Incorrect prediction (reward = 0.0)
        obs, info = env.reset()
        wrong_idx = (env.true_material_idx + 1) % 4
        obs, reward, done, trunc, info = env.step(wrong_idx)
        assert reward == 0.0
        assert info["correct"] is False

    def test_action_space_bounds(self):
        """Test action space has correct size."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)
        assert env.action_space.n == 4

        # Valid actions
        for action in range(4):
            env.reset()
            obs, reward, done, trunc, info = env.step(action)
            assert reward in [0.0, 1.0]

    def test_material_classes(self):
        """Test all material classes are present."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)
        assert env.MATERIAL_CLASSES == ["steel", "glass", "wood", "rubber"]

    def test_confusion_matrix_tracking(self):
        """Test confusion matrix is tracked correctly."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)

        # Make some predictions
        for _ in range(10):
            obs, info = env.reset()
            pred_idx = env.action_space.sample()
            env.step(pred_idx)

        # Check confusion matrix
        cm = env.get_confusion_matrix()
        assert cm.shape == (4, 4)
        assert cm.sum() == 10  # 10 predictions made

        # Reset and check
        env.reset_confusion_matrix()
        cm = env.get_confusion_matrix()
        assert cm.sum() == 0

    def test_observation_space_matches(self):
        """Test obs shape matches declared space."""
        env = SoundClassificationEnv(use_mel_spectrogram=True, hrtf_path=TEST_HRTF_PATH)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape

    def test_raw_audio_mode(self):
        """Test raw audio observation mode."""
        env = SoundClassificationEnv(
            use_mel_spectrogram=False,
            audio_duration=1.0,
            hrtf_path=TEST_HRTF_PATH
        )
        obs, _ = env.reset()
        assert obs.shape == (44100, 2)  # 1s stereo @ 44.1kHz
        assert obs.dtype == np.float32


class TestMultipleEpisodes:
    """Test multiple reset/step cycles."""

    def test_multiple_episodes(self):
        """Test environment can be reset multiple times."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)

        for _ in range(10):
            obs, info = env.reset()
            assert obs.shape == env.observation_space.shape

            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            assert done
            assert reward in [0.0, 1.0]

    def test_different_materials(self):
        """Test different episodes sample different materials."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)

        materials = []
        for _ in range(20):
            obs, info = env.reset()
            materials.append(info["true_material"])

        # Should have variety (not all the same)
        assert len(set(materials)) > 1


class TestAudioGeneration:
    """Test audio generation variations."""

    def test_all_materials_generate_audio(self):
        """Test audio generation works for all materials."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)

        # Sample many episodes to ensure all materials appear
        materials_seen = set()
        for _ in range(50):
            obs, info = env.reset()
            materials_seen.add(info["true_material"])
            assert obs.shape == env.observation_space.shape

        # Should have seen multiple materials
        assert len(materials_seen) >= 2

    def test_randomize_surface(self):
        """Test surface randomization."""
        env = SoundClassificationEnv(
            randomize_surface=True,
            hrtf_path=TEST_HRTF_PATH
        )

        surfaces = []
        for _ in range(20):
            obs, info = env.reset()
            surfaces.append(info["surface"])

        # Should have multiple surfaces
        assert len(set(surfaces)) > 1

    def test_fixed_position(self):
        """Test fixed position mode."""
        env = SoundClassificationEnv(
            randomize_position=False,
            hrtf_path=TEST_HRTF_PATH
        )

        positions = []
        azimuths = []
        for _ in range(5):
            obs, info = env.reset()
            positions.append(info["distance"])
            azimuths.append(info["azimuth"])

        # All should be same distance (2.0m) and azimuth (0.0°)
        assert all(d == 2.0 for d in positions)
        assert all(az == 0.0 for az in azimuths)

    def test_random_position(self):
        """Test random position mode."""
        env = SoundClassificationEnv(
            randomize_position=True,
            hrtf_path=TEST_HRTF_PATH
        )

        positions = []
        for _ in range(10):
            obs, info = env.reset()
            positions.append(info["distance"])

        # Should have variety in distances
        assert len(set(positions)) > 1


class TestSeedReproducibility:
    """Test seed reproducibility."""

    def test_reset_with_seed(self):
        """Test reset with seed produces same results."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)

        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)

        assert info1["true_material"] == info2["true_material"]
        assert np.allclose(obs1, obs2)

    def test_different_seeds(self):
        """Test different seeds produce different results."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)

        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=123)

        # Should likely be different (not guaranteed, but very likely)
        assert info1["true_material"] != info2["true_material"] or not np.allclose(obs1, obs2)


@pytest.mark.integration
class TestGymnasiumCompliance:
    """Test Gymnasium API compliance."""

    def test_gymnasium_check_env(self):
        """Test environment passes Gymnasium checks."""
        from gymnasium.utils.env_checker import check_env

        env = SoundClassificationEnv(
            audio_duration=0.5,
            hrtf_path=TEST_HRTF_PATH
        )
        check_env(env.unwrapped, skip_render_check=True)

    def test_reset_returns_tuple(self):
        """Test reset returns (obs, info) tuple."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_step_returns_five_tuple(self):
        """Test step returns 5-tuple."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)
        env.reset()
        result = env.step(env.action_space.sample())
        assert isinstance(result, tuple)
        assert len(result) == 5


@pytest.mark.integration
class TestInfoDict:
    """Test info dictionary contents."""

    def test_reset_info_contains_ground_truth(self):
        """Test reset info dict contains ground truth labels."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)
        obs, info = env.reset()

        required_keys = ["true_material", "true_material_idx", "surface", "source_pos", "distance", "azimuth"]
        for key in required_keys:
            assert key in info

    def test_step_info_contains_prediction(self):
        """Test step info contains prediction details."""
        env = SoundClassificationEnv(hrtf_path=TEST_HRTF_PATH)
        obs, info = env.reset()
        obs, reward, done, trunc, info = env.step(0)

        assert "predicted_material" in info
        assert "predicted_material_idx" in info
        assert "correct" in info
        assert info["predicted_material_idx"] == 0
        assert info["predicted_material"] == "steel"
