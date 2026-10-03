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
5. **Run Manual EDA:** Execute the preprocessing script to scale features and cache the data:
   ```bash
   python scripts/eda.py
   ```
   
## Running the Project
```bash
python src/main.py
```

Optional arguments (defaults shown):
```bash
python src/main.py --alpha 0.1 --lambda-reg 1.0 --test-size 0.2 --val-size 0.1
```
- `--alpha`: LinUCB exploration parameter.
- `--lambda-reg`: LinUCB ridge regularization parameter.
- `--test-size` / `--val-size`: proportions of the dataset for the test/validation splits (time-based).

## Evaluation
- Time-based 70/10/20 train/val/test split (no shuffling).
- Cost-sensitive reward matrix: `R_FP = -1`, `R_FN = -10`, `R_TN = R_TP = 0` (implied block threshold `p > 1/11`).
- Baseline decision thresholds are tuned on the validation split, then applied to the test set.
- LinUCB is warmed up sequentially on the train stream, then evaluated on the test stream.
- Metrics: cumulative reward, cost savings vs approve-all/block-all, confusion matrix, precision/recall, PR-AUC (probabilistic policies only).

### Current Results (default config)
| Policy | Cumulative Reward | Cost Savings vs Approve-All | Precision | Recall | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest (threshold 0.109) | -178 | +572 | 0.616 | 0.813 | 0.822 |
| LinUCB (alpha=0.1, lambda=1.0) | -191 | +559 | 0.838 | 0.760 | n/a |
| XGBoost (threshold 0.446) | -208 | +542 | 0.873 | 0.733 | 0.788 |
| Logistic Regression (threshold 0.980) | -218 | +532 | 0.469 | 0.800 | 0.748 |
| Approve All | -750 | 0 | 0.000 | 0.000 | n/a |
| Block All | -56887 | -56137 | 0.001 | 1.000 | n/a |

Test set: 56,962 transactions, 75 frauds. PR-AUC is `n/a` for policies without probabilistic output (LinUCB, naive baselines).