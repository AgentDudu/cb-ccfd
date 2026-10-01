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

## Evaluation
- Use time-based split or prequential evaluation.
- Compare cumulative reward/cost, cost savings vs approve-all/block-all, PR-AUC, precision/recall, and confusion matrix.
- Tune baseline thresholds to the same cost matrix used by the bandit.