"""Tests for reward functions."""

import pytest
import numpy as np

from physonic.envs.rewards import (
    angular_error,
    angular_error_reward,
    distance_reward,
    classification_reward
)


class TestAngularError:
    """Test angular error computation."""

    def test_zero_error(self):
        """Test no error when angles match."""
        assert angular_error(0.0, 0.0) == 0.0
        assert angular_error(45.0, 45.0) == 0.0
        assert angular_error(-90.0, -90.0) == 0.0

    def test_positive_error(self):
        """Test positive angular error."""
        assert angular_error(90.0, 0.0) == 90.0
        assert angular_error(0.0, 90.0) == 90.0

    def test_wraparound_error(self):
        """Test error computation with wraparound."""
        # 180° vs -180° should be 0° (same direction)
        assert angular_error(180.0, -180.0) == 0.0
        # 170° vs -170° should be 20° (not 340°)
        assert angular_error(170.0, -170.0) == pytest.approx(20.0)

    def test_maximum_error(self):
        """Test maximum error is 180°."""
        assert angular_error(0.0, 180.0) == 180.0
        assert angular_error(90.0, -90.0) == 180.0

    def test_symmetric_error(self):
        """Test error is symmetric."""
        error1 = angular_error(30.0, 60.0)
        error2 = angular_error(60.0, 30.0)
        assert error1 == error2


class TestAngularErrorReward:
    """Test angular error reward function."""

    def test_perfect_prediction(self):
        """Test reward is 0 for perfect prediction."""
        assert angular_error_reward(0.0, 0.0) == 0.0
        assert angular_error_reward(45.0, 45.0) == 0.0

    def test_worst_prediction(self):
        """Test reward is -1 for 180° error."""
        assert angular_error_reward(0.0, 180.0) == -1.0
        assert angular_error_reward(90.0, -90.0) == -1.0

    def test_90_degree_error(self):
        """Test reward for 90° error is -0.5."""
        reward = angular_error_reward(0.0, 90.0)
        assert reward == pytest.approx(-0.5)

    def test_reward_range(self):
        """Test reward is in [-1, 0] range."""
        for pred in np.linspace(-180, 180, 20):
            for true in np.linspace(-180, 180, 20):
                reward = angular_error_reward(pred, true)
                assert -1.0 <= reward <= 0.0


class TestDistanceReward:
    """Test distance reward function."""

    def test_zero_distance(self):
        """Test reward is 0 for perfect prediction."""
        pos = np.array([1.0, 2.0, 3.0])
        assert distance_reward(pos, pos) == 0.0

    def test_unit_distance(self):
        """Test reward for 1m error."""
        pred = np.array([0.0, 0.0, 0.0])
        true = np.array([1.0, 0.0, 0.0])
        assert distance_reward(pred, true) == pytest.approx(-1.0)

    def test_3d_distance(self):
        """Test 3D distance computation."""
        pred = np.array([0.0, 0.0, 0.0])
        true = np.array([1.0, 1.0, 1.0])
        expected = -np.sqrt(3)
        assert distance_reward(pred, true) == pytest.approx(expected)


class TestClassificationReward:
    """Test classification reward function."""

    def test_correct_classification(self):
        """Test reward is 1.0 for correct classification."""
        assert classification_reward(0, 0) == 1.0
        assert classification_reward(3, 3) == 1.0

    def test_incorrect_classification(self):
        """Test reward is 0.0 for incorrect classification."""
        assert classification_reward(0, 1) == 0.0
        assert classification_reward(3, 5) == 0.0
