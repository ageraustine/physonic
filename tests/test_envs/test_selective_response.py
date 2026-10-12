"""Tests for selective response environment."""

import pytest
import numpy as np

from physonic.envs import SelectiveResponseEnv

# Use test HRTF dataset for all tests
TEST_HRTF_PATH = "assets/hrtf/test_kemar/"


class TestSelectiveResponseEnv:
    """Test SelectiveResponseEnv basic functionality."""

    def test_env_creation(self):
        """Test environment can be created."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )
        assert env.action_space.n == 2  # IGNORE, RESPOND
        assert env.target_materials == ["glass"]
        assert set(env.distractor_materials) == {"steel", "wood", "rubber"}

    def test_invalid_target_material(self):
        """Test invalid target material raises error."""
        with pytest.raises(ValueError, match="Invalid target material"):
            SelectiveResponseEnv(
                target_materials=["invalid"],
                hrtf_path=TEST_HRTF_PATH
            )

    def test_all_targets_error(self):
        """Test error when all materials are targets."""
        with pytest.raises(ValueError, match="Must have at least one distractor"):
            SelectiveResponseEnv(
                target_materials=["steel", "glass", "wood", "rubber"],
                hrtf_path=TEST_HRTF_PATH
            )

    def test_reset_returns_valid_observation(self):
        """Test reset() returns valid observation."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )
        obs, info = env.reset()

        assert obs.shape == env.observation_space.shape
        assert "is_target" in info
        assert "material" in info
        assert info["is_target"] in [True, False]
        assert info["material"] in env.ALL_MATERIALS

    def test_step_computes_reward_correctly(self):
        """Test step() computes TP/TN/FP/FN rewards."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            reward_tp=1.0,
            reward_tn=0.5,
            reward_fp=-1.0,
            reward_fn=-2.0,
            hrtf_path=TEST_HRTF_PATH
        )

        # Force target material
        env.reset()
        env.is_target = True
        env.current_material = "glass"

        # True Positive: RESPOND to target
        obs, reward, done, trunc, info = env.step(1)
        assert reward == 1.0
        assert info["outcome"] == "TP"
        assert done is True

        # False Negative: IGNORE target
        env.reset()
        env.is_target = True
        env.current_material = "glass"
        obs, reward, done, trunc, info = env.step(0)
        assert reward == -2.0
        assert info["outcome"] == "FN"

        # True Negative: IGNORE distractor
        env.reset()
        env.is_target = False
        env.current_material = "steel"
        obs, reward, done, trunc, info = env.step(0)
        assert reward == 0.5
        assert info["outcome"] == "TN"

        # False Positive: RESPOND to distractor
        env.reset()
        env.is_target = False
        env.current_material = "steel"
        obs, reward, done, trunc, info = env.step(1)
        assert reward == -1.0
        assert info["outcome"] == "FP"

    def test_action_space_bounds(self):
        """Test action space has correct size."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )
        assert env.action_space.n == 2

        # Valid actions
        for action in [0, 1]:
            env.reset()
            obs, reward, done, trunc, info = env.step(action)
            assert info["outcome"] in ["TP", "TN", "FP", "FN"]

    def test_metrics_tracking(self):
        """Test precision/recall/F1 metrics."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )

        # Simulate perfect agent
        for _ in range(10):
            obs, info = env.reset()
            action = 1 if info["is_target"] else 0
            env.step(action)

        metrics = env.get_metrics()
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["accuracy"] == 1.0

    def test_confusion_matrix(self):
        """Test confusion matrix generation."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )

        # Run episodes
        for _ in range(20):
            obs, info = env.reset()
            action = env.action_space.sample()
            env.step(action)

        cm = env.get_confusion_matrix()
        assert cm.shape == (2, 2)
        assert cm.sum() == 20

    def test_reset_metrics(self):
        """Test metrics can be reset."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )

        # Run some episodes
        for _ in range(5):
            obs, info = env.reset()
            env.step(env.action_space.sample())

        # Reset metrics
        env.reset_metrics()

        metrics = env.get_metrics()
        assert metrics["tp"] == 0
        assert metrics["tn"] == 0
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0

    def test_target_probability(self):
        """Test target/distractor distribution."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            target_probability=0.5,
            hrtf_path=TEST_HRTF_PATH
        )

        targets = []
        for _ in range(100):
            obs, info = env.reset()
            targets.append(info["is_target"])

        # Should be roughly 50% targets with p=0.5
        target_ratio = sum(targets) / len(targets)
        assert 0.3 < target_ratio < 0.7  # Allow some variance

    def test_multiple_target_materials(self):
        """Test with multiple target materials."""
        env = SelectiveResponseEnv(
            target_materials=["glass", "steel"],
            hrtf_path=TEST_HRTF_PATH
        )

        assert set(env.target_materials) == {"glass", "steel"}
        assert set(env.distractor_materials) == {"wood", "rubber"}

        # Verify targets appear
        target_materials = []
        for _ in range(50):
            obs, info = env.reset()
            if info["is_target"]:
                target_materials.append(info["material"])

        # Should have both target materials
        assert len(set(target_materials)) >= 1  # At least one

    def test_observation_space_matches(self):
        """Test obs shape matches declared space."""
        env = SelectiveResponseEnv(
            use_mel_spectrogram=True,
            hrtf_path=TEST_HRTF_PATH
        )
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape

    def test_raw_audio_mode(self):
        """Test raw audio observation mode."""
        env = SelectiveResponseEnv(
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
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )

        for _ in range(10):
            obs, info = env.reset()
            assert obs.shape == env.observation_space.shape

            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            assert done
            assert info["outcome"] in ["TP", "TN", "FP", "FN"]

    def test_different_sounds(self):
        """Test different episodes sample different sounds."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )

        materials = []
        for _ in range(20):
            obs, info = env.reset()
            materials.append(info["material"])

        # Should have variety (not all the same)
        assert len(set(materials)) > 1


class TestPositionRandomization:
    """Test position randomization."""

    def test_fixed_position(self):
        """Test fixed position mode."""
        env = SelectiveResponseEnv(
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
        env = SelectiveResponseEnv(
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
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )

        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)

        assert info1["is_target"] == info2["is_target"]
        assert info1["material"] == info2["material"]
        assert np.allclose(obs1, obs2)

    def test_different_seeds(self):
        """Test different seeds produce different results."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )

        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=123)

        # Should likely be different (not guaranteed, but very likely)
        assert (info1["is_target"] != info2["is_target"] or
                info1["material"] != info2["material"] or
                not np.allclose(obs1, obs2))


@pytest.mark.integration
class TestGymnasiumCompliance:
    """Test Gymnasium API compliance."""

    def test_gymnasium_check_env(self):
        """Test environment passes Gymnasium checks."""
        from gymnasium.utils.env_checker import check_env

        env = SelectiveResponseEnv(
            target_materials=["glass"],
            audio_duration=0.5,
            hrtf_path=TEST_HRTF_PATH
        )
        check_env(env.unwrapped, skip_render_check=True)

    def test_reset_returns_tuple(self):
        """Test reset returns (obs, info) tuple."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_step_returns_five_tuple(self):
        """Test step returns 5-tuple."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )
        env.reset()
        result = env.step(env.action_space.sample())
        assert isinstance(result, tuple)
        assert len(result) == 5


@pytest.mark.integration
class TestInfoDict:
    """Test info dictionary contents."""

    def test_reset_info_contains_required_keys(self):
        """Test reset info dict contains required keys."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )
        obs, info = env.reset()

        required_keys = ["is_target", "material", "surface", "source_pos", "distance", "azimuth"]
        for key in required_keys:
            assert key in info

    def test_step_info_contains_outcome(self):
        """Test step info contains outcome details."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )
        obs, info = env.reset()
        obs, reward, done, trunc, info = env.step(0)

        assert "is_target" in info
        assert "agent_responds" in info
        assert "material" in info
        assert "outcome" in info
        assert "reward" in info
        assert info["agent_responds"] is False
        assert info["outcome"] in ["TN", "FN"]


class TestMetricsEdgeCases:
    """Test metrics computation edge cases."""

    def test_metrics_no_predictions(self):
        """Test metrics with no predictions."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            hrtf_path=TEST_HRTF_PATH
        )

        metrics = env.get_metrics()
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["accuracy"] == 0.0

    def test_metrics_only_negatives(self):
        """Test metrics when agent always ignores."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            target_probability=0.0,  # No targets
            hrtf_path=TEST_HRTF_PATH
        )

        for _ in range(10):
            env.reset()
            env.step(0)  # Always IGNORE

        metrics = env.get_metrics()
        assert metrics["precision"] == 0.0  # No TPs
        assert metrics["recall"] == 0.0  # No targets
        assert metrics["tn"] == 10

    def test_metrics_only_positives(self):
        """Test metrics when agent always responds."""
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            target_probability=1.0,  # All targets
            hrtf_path=TEST_HRTF_PATH
        )

        for _ in range(10):
            env.reset()
            env.step(1)  # Always RESPOND

        metrics = env.get_metrics()
        assert metrics["precision"] == 1.0  # All TPs
        assert metrics["recall"] == 1.0  # All targets caught
        assert metrics["tp"] == 10
