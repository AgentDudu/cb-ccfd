from typing import Dict, List
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
        alpha_decay: float = 1.0,
        r_fp: float = -1.0,
        r_fn: float = -10.0,
        r_tn: float = 0.0,
        r_tp: float = 0.0,
    ) -> None:
        """Initializes LinUCB model parameters for actions 0 and 1.

        Args:
            n_features: Dimensionality of context feature vectors.
            alpha: Initial exploration parameter controlling confidence bound width.
                Defaults to 1.0.
            lambda_reg: Ridge regularization parameter (prior). Defaults to 1.0.
            alpha_decay: Per-step geometric decay rate applied to alpha after each
                update (1.0 = no decay). Must satisfy 0 < alpha_decay <= 1.
                Defaults to 1.0.
            r_fp: Reward for blocking a legitimate transaction. Defaults to -1.0.
            r_fn: Reward for approving a fraudulent transaction. Defaults to -10.0.
            r_tn: Reward for approving a legitimate transaction. Defaults to 0.0.
            r_tp: Reward for blocking a fraudulent transaction. Defaults to 0.0.

        Raises:
            ValueError: If the reward matrix cannot be inverted to recover a fraud
                probability estimate (both arms degenerate).
        """
        self.n_features = n_features
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.t = 0
        self.lambda_reg = lambda_reg
        self.r_fp, self.r_fn, self.r_tn, self.r_tp = r_fp, r_fn, r_tn, r_tp

        self._denominators: Dict[int, float] = {0: r_fn - r_tn, 1: r_tp - r_fp}
        if all(den == 0.0 for den in self._denominators.values()):
            raise ValueError(
                "Reward matrix must vary with the label in at least one arm to "
                "recover a fraud probability."
            )

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
        alpha_t = self._current_alpha()

        mean_0 = X_arr @ self.theta[0]
        var_0 = np.sum((X_arr @ self.A_inv[0]) * X_arr, axis=1)
        p_0 = mean_0 + alpha_t * np.sqrt(np.maximum(var_0, 0.0))

        mean_1 = X_arr @ self.theta[1]
        var_1 = np.sum((X_arr @ self.A_inv[1]) * X_arr, axis=1)
        p_1 = mean_1 + alpha_t * np.sqrt(np.maximum(var_1, 0.0))

        return np.where(p_1 > p_0, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimates the fraud probability implied by the fitted arm models.

        Each arm model estimates the expected reward `q_a = theta_a^T x`, which for a
        fraud probability `p = P(y = 1 | x)` is the linear mixture

            q_0 = R_TN * (1 - p) + R_FN * p
            q_1 = R_FP * (1 - p) + R_TP * p

        Inverting each well-defined arm gives two independent estimates of `p`, which
        are averaged and clipped to [0, 1]. This is a post-hoc readout of the linear
        reward estimates: unlike GLM-UCB there is no probabilistic model, so the
        resulting scores are only rank-informative, not calibrated.

        Args:
            X: 2D array of context feature vectors of shape (n_samples, n_features).

        Returns:
            1D array of implied fraud probability scores in [0, 1].
        """
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        estimates: List[np.ndarray] = []
        offsets = {0: self.r_tn, 1: self.r_fp}

        for action, den in self._denominators.items():
            if den != 0.0:
                estimates.append((X_arr @ self.theta[action] - offsets[action]) / den)

        return np.clip(np.mean(estimates, axis=0), 0.0, 1.0)

    def _current_alpha(self) -> float:
        """Computes the current exploration parameter after per-step decay."""
        return self.alpha * self.alpha_decay ** self.t

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

        self.t += 1
