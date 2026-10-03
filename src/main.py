import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processing.pipeline import load_and_split_data
from src.evaluation.metrics import (
    calculate_confusion_matrix,
    calculate_cost_savings,
    calculate_cumulative_reward,
    calculate_pr_auc,
    calculate_precision_recall,
    find_optimal_threshold,
)
from src.models.baselines import train_and_predict_baselines
from src.models.linucb import LinUCB


def print_policy_report(
    name: str,
    y_true: np.ndarray,
    actions: np.ndarray,
    probs: Optional[np.ndarray],
    threshold: Optional[float],
    approve_all_reward: float,
    block_all_reward: float,
) -> None:
    """Prints the evaluation report for a single policy.

    Args:
        name: Display name of the policy.
        y_true: Ground truth binary target labels.
        actions: Predicted binary actions of the policy.
        probs: Predicted fraud probabilities, if the policy is probabilistic.
        threshold: Decision threshold used, if any.
        approve_all_reward: Cumulative reward of the approve-all policy.
        block_all_reward: Cumulative reward of the block-all policy.
    """
    reward = calculate_cumulative_reward(y_true, actions)
    savings_approve_all, savings_block_all = calculate_cost_savings(
        reward, approve_all_reward, block_all_reward
    )
    cm = calculate_confusion_matrix(y_true, actions)
    precision, recall = calculate_precision_recall(y_true, actions)
    pr_auc = calculate_pr_auc(y_true, probs) if probs is not None else None

    print(f"Policy: {name}")
    if threshold is not None:
        print(f"  Decision Threshold: {threshold:.4f}")
    print(f"  Cumulative Reward: {reward:.2f}")
    print(f"  Cost Savings vs Approve All: {savings_approve_all:.2f}")
    print(f"  Cost Savings vs Block All: {savings_block_all:.2f}")
    print(
        f"  Confusion Matrix: TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}"
    )
    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f}")
    if pr_auc is not None:
        print(f"  PR-AUC: {pr_auc:.4f}")
    else:
        print("  PR-AUC: n/a (no probabilistic output)")


def run_bandit_stream(bandit: LinUCB, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Runs the bandit sequentially over a stream of contexts, choosing and updating actions.

    Args:
        bandit: LinUCB model to run.
        X: 2D array of context feature vectors of shape (n_samples, n_features).
        y: Ground truth binary target labels for the stream.

    Returns:
        1D array of chosen actions (0 or 1) for each context in the stream.
    """
    actions = np.empty(X.shape[0], dtype=np.int64)

    for i in range(X.shape[0]):
        x_t = X[i]
        y_t = int(y[i])
        action = int(bandit.predict(x_t)[0])
        actions[i] = action
        bandit.update(x_t, action, calculate_cumulative_reward(y_t, action))

    return actions


def main() -> None:
    """Executes the full pipeline: data loading, bandit evaluation, and baseline comparison."""
    # 1. Call the data pipeline
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data()

    X_test_arr = np.asarray(X_test, dtype=np.float64)
    y_val_arr = np.asarray(y_val, dtype=np.int64)
    y_test_arr = np.asarray(y_test, dtype=np.int64)

    # 2. Warm up the LinUCB on the train stream, then run it sequentially on the test set
    n_features = X_train.shape[1]
    bandit = LinUCB(n_features=n_features, alpha=0.1, lambda_reg=1.0)

    X_train_arr = np.asarray(X_train, dtype=np.float64)
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    run_bandit_stream(bandit, X_train_arr, y_train_arr)

    bandit_actions = run_bandit_stream(bandit, X_test_arr, y_test_arr)

    # 3. Run the baselines: tune the threshold on the validation set, apply to test
    baseline_probs = train_and_predict_baselines(X_train, y_train, X_val, X_test)
    baseline_reports = []
    for name, (val_probs, test_probs) in baseline_probs.items():
        opt_thresh = find_optimal_threshold(y_val_arr, val_probs)
        preds = (test_probs >= opt_thresh).astype(int)
        baseline_reports.append((name, preds, test_probs, opt_thresh))

    # Naive baselines
    approve_all_actions = np.zeros_like(y_test_arr)
    block_all_actions = np.ones_like(y_test_arr)

    approve_all_reward = calculate_cumulative_reward(y_test_arr, approve_all_actions)
    block_all_reward = calculate_cumulative_reward(y_test_arr, block_all_actions)

    # 4. Print the evaluation report
    policies: List[Tuple[str, np.ndarray, Optional[np.ndarray], Optional[float]]] = [
        ("LinUCB", bandit_actions, None, None),
        *baseline_reports,
        ("Approve All", approve_all_actions, None, None),
        ("Block All", block_all_actions, None, None),
    ]

    print("=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    print(f"Test Set Size: {len(y_test_arr)}")

    for name, actions, probs, threshold in policies:
        print_policy_report(
            name, y_test_arr, actions, probs, threshold, approve_all_reward, block_all_reward
        )


if __name__ == "__main__":
    main()
