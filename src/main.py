import sys
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processing.pipeline import load_and_split_data
from src.evaluation.metrics import calculate_cumulative_reward, find_optimal_threshold
from src.models.baselines import train_and_predict_baselines
from src.models.linucb import LinUCB


def main() -> None:
    """Executes the full pipeline: data loading, bandit evaluation, and baseline comparison."""
    # 1. Call the data pipeline
    X_train, y_train, X_test, y_test = load_and_split_data()

    X_test_arr = np.asarray(X_test, dtype=np.float64)
    y_test_arr = np.asarray(y_test, dtype=np.int64)

    # 2. Run the LinUCB model sequentially on the test set
    n_features = X_train.shape[1]
    bandit = LinUCB(n_features=n_features, alpha=0.1, lambda_reg=1.0)
    bandit_cumulative_reward = 0.0

    for i in range(len(X_test_arr)):
        x_t = X_test_arr[i]
        y_t = int(y_test_arr[i])
        action = int(bandit.predict(x_t)[0])
        reward = calculate_cumulative_reward(y_t, action)
        bandit.update(x_t, action, reward)
        bandit_cumulative_reward += reward

    # 3. Run the baselines on the test set using the optimal threshold
    baseline_probs = train_and_predict_baselines(X_train, y_train, X_test)
    baseline_rewards = {}

    for name, probs in baseline_probs.items():
        opt_thresh = find_optimal_threshold(y_test_arr, probs)
        preds = (probs >= opt_thresh).astype(int)
        baseline_rewards[name] = calculate_cumulative_reward(y_test_arr, preds)

    best_baseline_name = max(baseline_rewards, key=baseline_rewards.get)
    best_baseline_reward = baseline_rewards[best_baseline_name]

    # Naive baselines
    approve_all_preds = np.zeros_like(y_test_arr)
    approve_all_reward = calculate_cumulative_reward(y_test_arr, approve_all_preds)

    block_all_preds = np.ones_like(y_test_arr)
    block_all_reward = calculate_cumulative_reward(y_test_arr, block_all_preds)

    # 4. Print cumulative rewards
    print(f"Bandit Cumulative Reward: {bandit_cumulative_reward:.2f}")
    print(
        f"Best Baseline ({best_baseline_name}) Cumulative Reward: {best_baseline_reward:.2f}"
    )
    print(f"Approve All Baseline Cumulative Reward: {approve_all_reward:.2f}")
    print(f"Block All Baseline Cumulative Reward: {block_all_reward:.2f}")


if __name__ == "__main__":
    main()
