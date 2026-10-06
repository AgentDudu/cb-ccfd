# Credit Card Fraud Detection using Contextual Bandits

This project implements a Contextual Bandit model for credit card fraud detection and compares its performance against traditional supervised classification algorithms (e.g., Random Forest, XGBoost, Logistic Regression). 

## Dataset
The project uses the [ULB Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). 
**Note:** The dataset is not included in this repository. Please download `creditcard.csv` and place it in the `data/raw/` directory.

## Workflow
This project uses a simplified, single-branch workflow directly on the `main` branch to maximize development speed.

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it and install dependencies: `pip install -r requirements.txt`
4. Place `creditcard.csv` in `data/raw/`.
5. **Run Manual EDA:** Execute the preprocessing script to clean and cache the data (features are left unscaled; scaling is fitted on the train split inside the data pipeline):
   ```bash
   python scripts/eda.py
   ```
   
## Running the Project
```bash
python src/main.py
```

Optional arguments (defaults shown):
```bash
python src/main.py --models linucb,glmucb --alpha 0.1 --lambda-reg 1.0 --alpha-decay 1.0 \
  --glm-alpha 0.05 --glm-min-curvature 1e-5 --test-size 0.2 --val-size 0.1
```
- `--models`: Comma-separated bandit policies to evaluate (`linucb`, `glmucb`).
- `--alpha`: LinUCB exploration parameter.
- `--lambda-reg`: LinUCB ridge regularization parameter.
- `--alpha-decay`: LinUCB per-step alpha decay rate (1.0 = no decay).
- `--glm-alpha`: GLM-UCB exploration parameter, on the fraud-probability scale.
- `--glm-min-curvature`: GLM-UCB Hessian curvature floor; the main exploration
  knob (see *Model variants*).
- `--test-size` / `--val-size`: proportions of the dataset for the test/validation splits (time-based).

## Evaluation
- Time-based 70/10/20 train/val/test split (no shuffling).
- Cost-sensitive reward matrix: `R_FP = -1`, `R_FN = -10`, `R_TN = R_TP = 0` (implied block threshold `p > 1/11`).
- Baseline decision thresholds are tuned on the validation split, then applied to the test set.
- LinUCB is warmed up sequentially on the train stream, then evaluated on the test stream.
  GLM-UCB follows the same warm-up/eval protocol.
- Metrics: cumulative reward, cost savings vs approve-all/block-all, confusion matrix, precision/recall, PR-AUC (probabilistic policies only).

### Model variants
**LinUCB** (`src/models/linucb.py`) assumes the expected reward is linear in the context.
**GLM-UCB** (`src/models/glmucb.py`) relaxes that (see `MATH_MODEL.md` sec. 3, "Linearity"):
it fits a single logistic GLM for the fraud probability `p(x) = P(y=1|x)` by online
(one-step) Newton/IRLS, applies optimism to the *probability*, and then derives the action
from the reward matrix instead of from per-arm reward estimates:

```
p_bar(x) = clip(sigmoid(x^T theta) + alpha * sqrt(x^T H^-1 x), 0, 1)
block  iff  p_bar(x) > p*,   p* = -R_FP / (-R_FN - R_FP) = 1/11
H = lambda*I + sum_t sigma'(x_t^T theta_t) x_t x_t^T
```

Bandit feedback is preserved: only the chosen action's reward is observed, and the label is
recovered from that reward given the action.

Implementation notes from tuning:
- A naive Faury-style *per-arm* logistic model of the rescaled reward degenerates here:
  the reward matrix (not the context) drives a constant ~0.1 offset between the two arm
  means, so the confidence bonus dominates and the policy collapses to block-all.
- `min_curvature` is the dominant exploration knob. Rare-event curvature is ~1e-6; with a
  1e-3 floor the Hessian grows too fast, the ellipsoid collapses (mean UCB bonus ~0.004 at
  alpha=0.1) and exploration is effectively disabled. 1e-5 keeps a usable width after 200k+
  updates. Reward degrades sharply beyond alpha ~0.3 (over-blocking) and with `lambda_reg`
  well below 1.0 (online Newton becomes unstable, PR-AUC drops to ~0.33).
- GLM-UCB models probabilities directly, and LinUCB now exposes a post-hoc `predict_proba`
  that inverts the reward matrix (`p = (q_a - R(a,0)) / (R(a,1) - R(a,0))`, averaged over both
  arms), so PR-AUC is reported for both bandits.

### Current Results (default config)
| Policy | Cumulative Reward | Cost Savings vs Approve-All | Precision | Recall | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest (threshold 0.109) | -178 | +572 | 0.616 | 0.813 | 0.822 |
| LinUCB (alpha=0.1, lambda=1.0) | -192 | +558 | 0.826 | 0.760 | 0.202 |
| GLM-UCB (alpha=0.05, lambda=1.0, min_curv=1e-5) | -202 | +548 | 0.824 | 0.747 | 0.720 |
| XGBoost (threshold 0.446) | -208 | +542 | 0.873 | 0.733 | 0.788 |
| Logistic Regression (threshold 0.980) | -219 | +531 | 0.465 | 0.800 | 0.745 |
| Approve All | -750 | 0 | 0.000 | 0.000 | n/a |
| Block All | -56887 | -56137 | 0.001 | 1.000 | n/a |

Test set: 56,962 transactions, 75 frauds. PR-AUC is `n/a` only for the deterministic naive
baselines (approve-all / block-all).
GLM-UCB ranks below LinUCB on cumulative reward but above XGBoost and Logistic Regression.
On *ranking* quality the order reverses: GLM-UCB's PR-AUC (0.720) is more than 3x LinUCB's
implied-probability PR-AUC (0.202). LinUCB picks its actions well at the linear-UCB margin
but its fitted arm models order frauds poorly, which is consistent with the linearity
misspecification discussed in `MATH_MODEL.md` sec. 3.