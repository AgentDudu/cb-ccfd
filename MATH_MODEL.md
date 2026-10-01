# Mathematical Model: Contextual Bandit for Fraud Classification

We frame fraud detection as a sequential, cost-sensitive decision problem under bandit feedback.

## 1. Problem Formulation
At each time step $t$ (a transaction):
1. **Context ($x_t$):** Feature vector $x_t \in \mathbb{R}^d$.
2. **Action ($a_t$):** Agent selects $a_t \in \{0,1\}$:
   - $0$ = Approve / Legitimate
   - $1$ = Block / Decline
3. **Reward ($r_t$):** True label $y_t \in \{0,1\}$ ($1$ = fraud). Reward is
   $$
   r_t = R(a_t, y_t)
   $$
   and is observed only for the chosen action $a_t$. In delayed/censored settings, $y_t$ may be revealed later.

**Note:** If full labels are always available for both actions, the problem is full-information, not a bandit. This implementation assumes bandit feedback.

## 2. Cost-Sensitive Reward Function
Define $R(a, y)$:

$$
r_t = \begin{cases}
R_{TN} & a_t=0, y_t=0 \\
R_{FP} & a_t=1, y_t=0 \quad \text{(e.g., } -1.0) \\
R_{FN} & a_t=0, y_t=1 \quad \text{(e.g., } -10.0) \\
R_{TP} & a_t=1, y_t=1 \quad \text{(e.g., } 0.0)
\end{cases}
$$

Objective: maximize $\mathbb{E}[\sum_{t=1}^T r_t]$.

If $R_{TN}=R_{TP}=0$, the implied fraud probability threshold to choose action $a=1$ (Block) over $a=0$ (Approve) is:
$$
p > \frac{-R_{FP}}{-R_{FN} - R_{FP}}
$$
For $R_{FP}=-1$, $R_{FN}=-10$, this is $p > 1/11 \approx 9.09\%$.

## 3. Algorithm: LinUCB
For each action $a \in \{0,1\}$, maintain:
- $A_a = \lambda I_d$ (ridge prior, $\lambda > 0$)
- $b_a = 0 \in \mathbb{R}^d$

### Parameter Estimation
$$
\hat{\theta}_a = A_a^{-1} b_a
$$

### Action Selection
$$
p_{t,a} = \hat{\theta}_a^T x_t + \alpha \sqrt{x_t^T A_a^{-1} x_t}
$$
$$
a_t = \arg\max_{a \in \{0,1\}} p_{t,a}
$$

Tie-breaking: random or prefer $a=0$ (approve).

### Model Update
Update only the chosen action:
$$
A_{a_t} \leftarrow A_{a_t} + x_t x_t^T
$$
$$
b_{a_t} \leftarrow b_{a_t} + r_t x_t
$$

### Practical Notes
- **Efficiency:** Use Cholesky or Sherman–Morrison rank-1 updates; avoid explicit inverse.
- **Feature scaling:** Standardize features. The UCB term is scale-sensitive.
- **Linearity:** LinUCB assumes linear expected reward. If $P(y=1|x)$ is logistic, consider GLM-UCB / logistic UCB or justify a linear probability model.
- **Exploration:** $\alpha$ controls exploration. In production, use conservative or decaying $\alpha$ to limit costly exploration.

## 4. Comparison with Baseline Classifiers
Baselines (Logistic Regression, Random Forest, etc.) must be cost-sensitive:
- Use class weights.
- Tune decision threshold to the same reward matrix.
- Evaluate on the same sequential stream with the same reward matrix.

### Evaluation Metrics
- Cumulative reward / cost
- Cost savings vs approve-all and block-all
- PR-AUC, precision/recall at chosen threshold, confusion matrix
- Time-based split or prequential evaluation to avoid leakage

**Key difference:** Baselines are batch, i.i.d., and need threshold tuning. The bandit updates incrementally, handles exploration–exploitation, and directly optimizes asymmetric business cost.