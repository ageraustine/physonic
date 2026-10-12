"""Reward functions for acoustic perception tasks."""

import numpy as np


def angular_error(pred_deg: float, true_deg: float) -> float:
    """
    Compute shortest angular distance between two angles.

    Args:
        pred_deg: Predicted angle in degrees
        true_deg: True angle in degrees

    Returns:
        Angular error in [0, 180] degrees
    """
    # Normalize to [-180, 180]
    diff = (pred_deg - true_deg + 180) % 360 - 180
    return abs(diff)


def angular_error_reward(pred_deg: float, true_deg: float) -> float:
    """
    Reward based on angular error.

    Returns:
        Reward in [-1, 0] where 0 is perfect prediction
    """
    error = angular_error(pred_deg, true_deg)
    return -error / 180.0


def distance_reward(pred_pos: np.ndarray, true_pos: np.ndarray) -> float:
    """
    Reward based on Euclidean distance error.

    Args:
        pred_pos: Predicted (x, y, z) position
        true_pos: True (x, y, z) position

    Returns:
        Negative L2 distance
    """
    distance = np.linalg.norm(pred_pos - true_pos)
    return -distance


def classification_reward(pred_bin: int, true_bin: int) -> float:
    """
    Binary reward for discrete bin classification.

    Args:
        pred_bin: Predicted azimuth bin index
        true_bin: True azimuth bin index

    Returns:
        1.0 if correct, 0.0 if wrong
    """
    return 1.0 if pred_bin == true_bin else 0.0
