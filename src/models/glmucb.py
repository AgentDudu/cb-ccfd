from typing import Optional
import numpy as np


class GLMUCB:
    """Logistic (GLM-)UCB contextual bandit for cost-sensitive fraud decisions.

    LinUCB assumes the expected reward is *linear* in the context, which is
    misspecified here: the reward is a piecewise-linear function of the fraud
    probability `p(x) = P(y = 1 | x)`, and `p(x)` is logistic in the VSA
    features (see `MATH_MODEL.md` sec. 3, "Linearity").

    This policy keeps the bandit structure but replaces the linear arm models
    with a single logistic GLM for `p(x)`, estimated by online (one-step)
    Newton / IRLS minimization of the regularized log-loss

        L(theta) = sum_t -[ y_t log(sigma(x_t^T theta))
                            + (1 - y_t) log(1 - sigma(x_t^T theta)) ]
                   + (lambda / 2) * ||theta||^2 .

    Optimism is applied to the *probability*, and the action is then chosen
    from the cost-sensitive reward matrix rather than from per-arm reward
    estimates:

        p_bar_t(x) = clip(sigma(x^T theta) + alpha * sqrt(x^T H^-1 x), 0, 1)
        a_t = 1 (block)  iff  p_bar_t(x) > p*,
        p*  = -R_FP / (-R_FN - R_FP)   (= 1/11 for R_FP = -1, R_FN = -10)

    where `H` is the regularized logistic Hessian, `H = lambda*I + sum_t
    sigma'(x_t^T theta_t) x_t x_t^T`. This is equivalent to comparing the
    optimistic expected rewards `R(a, y)` under `p_bar`, and it avoids the
    degeneracy of fitting one logistic model per arm: because the reward matrix
    is action-dependent while the label is not, per-arm reward models converge
    to two constants that differ only by a reward-matrix offset, which makes the
    confidence bonus dominate and collapses the policy to block-all.

    Bandit feedback is respected: only the chosen action's reward is observed,
    and the label is recovered from that reward given the action.
    """

    def __init__(
        self,
        n_features: int,
        alpha: float = 0.05,
        lambda_reg: float = 1.0,
        alpha_decay: float = 1.0,
        r_fp: float = -1.0,
        r_fn: float = -10.0,
        r_tn: float = 0.0,
        r_tp: float = 0.0,
        min_curvature: float = 1e-5,
        symmetrize_every: int = 5000,
    ) -> None:
        """Initializes the logistic-UCB model and its reward matrix.

        Args:
            n_features: Dimensionality of context feature vectors.
            alpha: Exploration parameter on the fraud-probability scale
                (0..1). Acts as a soft threshold shift: the policy blocks when
                `p_hat > p* - alpha * sqrt(x^T H^-1 x)`. Defaults to 0.05.
            lambda_reg: Ridge regularization parameter (prior). Defaults to 1.0.
                Values much below 1.0 destabilize the online Newton estimate on
                this dataset.
            alpha_decay: Per-step geometric decay rate applied to alpha after
                each update (1.0 = no decay). Must satisfy
                0 < alpha_decay <= 1. Defaults to 1.0.
            r_fp: Reward for blocking a legitimate transaction. Defaults to -1.0.
            r_fn: Reward for approving a fraudulent transaction. Defaults to -10.0.
            r_tn: Reward for approving a legitimate transaction. Defaults to 0.0.
            r_tp: Reward for blocking a fraudulent transaction. Defaults to 0.0.
            min_curvature: Lower bound on the logistic curvature sigma'(x^T theta)
                when accumulating the Hessian. This is the key exploration knob:
                rare-event curvature is ~1e-6, so a large floor (e.g. 1e-3)
                shrinks the ellipsoid too fast and disables exploration, while
                the default 1e-5 keeps a usable confidence width after 200k+
                updates. Defaults to 1e-5.
            symmetrize_every: Number of updates between numerical
                re-symmetrizations of the stored inverse Hessian. Defaults to 5000.

        Raises:
            ValueError: If the reward matrix does not define a valid block
                threshold in (0, 1).
        """
        denom = -(r_fn - r_fp)
        if denom <= 0:
            raise ValueError(
                "Reward matrix must satisfy R_FN < R_FP to define a block threshold."
            )
        self.block_threshold = -r_fp / denom

        self.n_features = n_features
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.lambda_reg = lambda_reg
        self.min_curvature = min_curvature
        self.symmetrize_every = symmetrize_every
        self.t = 0

        self._reward_lookup = {(0, r_tn): 0, (0, r_fn): 1, (1, r_fp): 0, (1, r_tp): 1}

        self.theta: np.ndarray = np.zeros(n_features, dtype=np.float64)
        self.H_inv: np.ndarray = (1.0 / lambda_reg) * np.eye(
            n_features, dtype=np.float64
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Selects actions for given contexts using the optimistic threshold rule.

        Args:
            X: 2D array of context feature vectors of shape (n_samples, n_features).

        Returns:
            1D array of chosen actions (0 = approve, 1 = block) for each sample.
        """
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        p_bar = self.predict_ucb(X_arr)
        return np.where(p_bar > self.block_threshold, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns the model's point estimate of the fraud probability.

        Args:
            X: 2D array of context feature vectors of shape (n_samples, n_features).

        Returns:
            1D array of estimated fraud probabilities P(y = 1 | x).
        """
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return self._sigmoid(X_arr @ self.theta)

    def predict_ucb(self, X: np.ndarray) -> np.ndarray:
        """Returns the upper confidence bound on the fraud probability.

        Args:
            X: 2D array of context feature vectors of shape (n_samples, n_features).

        Returns:
            1D array of optimistic fraud probabilities, clipped to [0, 1].
        """
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        p = self._sigmoid(X_arr @ self.theta)
        var = np.sum((X_arr @ self.H_inv) * X_arr, axis=1)
        bonus = self._current_alpha() * np.sqrt(np.maximum(var, 0.0))
        return np.clip(p + bonus, 0.0, 1.0)

    def update(self, x: np.ndarray, action: int, reward: float) -> None:
        """Applies one online Newton step using the observed bandit reward.

        The ground-truth label is recovered from the observed reward and the
        action taken, so no counterfactual information is used.

        Args:
            x: Context vector of shape (n_features,).
            action: Chosen action (0 = approve, 1 = block).
            reward: Observed cost-sensitive reward for the chosen action.

        Raises:
            ValueError: If the reward is inconsistent with the action taken.
        """
        x_vec = np.asarray(x, dtype=np.float64).ravel()
        y = self._label_from_reward(action, reward)

        p = float(self._sigmoid(np.dot(x_vec, self.theta)))
        curvature = max(p * (1.0 - p), self.min_curvature)

        v = self.H_inv @ x_vec
        denom = 1.0 + curvature * float(np.dot(x_vec, v))
        self.H_inv -= (curvature / denom) * np.outer(v, v)

        self.theta = self.theta + self.H_inv @ ((y - p) * x_vec)

        self.t += 1
        if self.symmetrize_every and self.t % self.symmetrize_every == 0:
            self.H_inv = 0.5 * (self.H_inv + self.H_inv.T)

    def _label_from_reward(self, action: int, reward: float) -> int:
        """Recovers the true label from the observed reward and chosen action.

        Args:
            action: Chosen action (0 = approve, 1 = block).
            reward: Observed reward for that action.

        Returns:
            The recovered label (0 = legitimate, 1 = fraud).

        Raises:
            ValueError: If the reward does not match any outcome of that action.
        """
        exact = self._reward_lookup.get((action, reward))
        if exact is not None:
            return exact
        for (act, r), label in self._reward_lookup.items():
            if act == action and np.isclose(reward, r):
                return label
        raise ValueError(f"Reward {reward} is inconsistent with action {action}.")

    def _current_alpha(self) -> float:
        """Computes the current exploration parameter after per-step decay."""
        return self.alpha * self.alpha_decay ** self.t

    @staticmethod
    def _sigmoid(u: np.ndarray) -> np.ndarray:
        """Numerically stable logistic sigmoid.

        Args:
            u: Pre-activation values.

        Returns:
            Sigmoid of the input, elementwise.
        """
        pos = u >= 0
        exp_neg = np.exp(np.where(pos, -np.clip(u, 0.0, 709.0), 0.0))
        exp_pos = np.exp(np.where(pos, 0.0, np.clip(u, -709.0, 0.0)))
        return np.where(pos, 1.0 / (1.0 + exp_neg), exp_pos / (1.0 + exp_pos))
