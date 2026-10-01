from typing import Union
import numpy as np
import pandas as pd


def calculate_cumulative_reward(
    y_true: Union[pd.Series, np.ndarray, int],
    y_pred: Union[pd.Series, np.ndarray, int],
    r_fp: float = -1.0,
    r_fn: float = -10.0,
    r_tn: float = 0.0,
    r_tp: float = 0.0,
) -> float:
    """Calculates cumulative reward based on the cost-sensitive reward matrix.

    Args:
        y_true: Ground truth binary target labels (0 for legitimate, 1 for fraud).
        y_pred: Predicted actions (0 for approve, 1 for block).
        r_fp: Reward for false positive (blocking legitimate). Defaults to -1.0.
        r_fn: Reward for false negative (approving fraud). Defaults to -10.0.
        r_tn: Reward for true negative (approving legitimate). Defaults to 0.0.
        r_tp: Reward for true positive (blocking fraud). Defaults to 0.0.

    Returns:
        Cumulative reward value.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    fp = np.sum((y_t == 0) & (y_p == 1))
    fn = np.sum((y_t == 1) & (y_p == 0))
    tn = np.sum((y_t == 0) & (y_p == 0))
    tp = np.sum((y_t == 1) & (y_p == 1))

    return float(fp * r_fp + fn * r_fn + tn * r_tn + tp * r_tp)


def find_optimal_threshold(
    y_true: Union[pd.Series, np.ndarray],
    y_probs: Union[pd.Series, np.ndarray],
    num_thresholds: int = 100,
) -> float:
    """Finds the probability threshold that maximizes cumulative reward.

    Args:
        y_true: Ground truth binary target labels.
        y_probs: Predicted probabilities for the positive class (fraud).
        num_thresholds: Number of candidate thresholds to evaluate between 0 and 1.
            Defaults to 100.

    Returns:
        The probability threshold that maximizes the cumulative reward.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_probs)

    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_threshold = 0.5
    best_reward = float("-inf")

    for thresh in thresholds:
        preds = (y_p >= thresh).astype(int)
        reward = calculate_cumulative_reward(y_t, preds)
        if reward > best_reward:
            best_reward = reward
            best_threshold = float(thresh)

    return best_threshold
