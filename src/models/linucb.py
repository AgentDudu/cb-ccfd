from typing import Dict
import numpy as np


class LinUCB:
    """LinUCB contextual bandit algorithm for binary action spaces.

    Implements linear upper confidence bound exploration with ridge regression
    and Sherman-Morrison rank-1 updates for action selection and parameter
    estimation.
    """

    def __init__(
        self,
        n_features: int,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
    ) -> None:
        """Initializes LinUCB model parameters for actions 0 and 1.

        Args:
            n_features: Dimensionality of context feature vectors.
            alpha: Exploration parameter controlling confidence bound width.
                Defaults to 1.0.
            lambda_reg: Ridge regularization parameter (prior). Defaults to 1.0.
        """
        self.n_features = n_features
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        self.A: Dict[int, np.ndarray] = {
            0: lambda_reg * np.eye(n_features, dtype=np.float64),
            1: lambda_reg * np.eye(n_features, dtype=np.float64),
        }
        self.b: Dict[int, np.ndarray] = {
            0: np.zeros(n_features, dtype=np.float64),
            1: np.zeros(n_features, dtype=np.float64),
        }
        self.A_inv: Dict[int, np.ndarray] = {
            0: (1.0 / lambda_reg) * np.eye(n_features, dtype=np.float64),
            1: (1.0 / lambda_reg) * np.eye(n_features, dtype=np.float64),
        }
        self.theta: Dict[int, np.ndarray] = {
            0: np.zeros(n_features, dtype=np.float64),
            1: np.zeros(n_features, dtype=np.float64),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Selects actions for given contexts using the UCB formula.

        Args:
            X: 2D array of context feature vectors of shape (n_samples, n_features).

        Returns:
            1D array of chosen actions (0 or 1) for each sample.
        """
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))

        mean_0 = X_arr @ self.theta[0]
        var_0 = np.sum((X_arr @ self.A_inv[0]) * X_arr, axis=1)
        p_0 = mean_0 + self.alpha * np.sqrt(np.maximum(var_0, 0.0))

        mean_1 = X_arr @ self.theta[1]
        var_1 = np.sum((X_arr @ self.A_inv[1]) * X_arr, axis=1)
        p_1 = mean_1 + self.alpha * np.sqrt(np.maximum(var_1, 0.0))

        return np.where(p_1 > p_0, 1, 0)

    def update(self, x: np.ndarray, action: int, reward: float) -> None:
        """Updates A and b for the specific action using Sherman-Morrison rank-1 update.

        Args:
            x: Context vector of shape (n_features,).
            action: Chosen action (0 or 1).
            reward: Observed reward for the chosen action.
        """
        x_vec = np.asarray(x, dtype=np.float64).ravel()

        self.A[action] += np.outer(x_vec, x_vec)
        self.b[action] += reward * x_vec

        v = self.A_inv[action] @ x_vec
        denom = 1.0 + np.dot(x_vec, v)
        self.A_inv[action] -= np.outer(v, v) / denom
        self.theta[action] = self.A_inv[action] @ self.b[action]
