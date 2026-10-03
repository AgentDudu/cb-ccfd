from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_and_predict_baselines(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Trains baseline classifiers with class imbalance handling and predicts fraud probabilities.

    Args:
        X_train: Training feature matrix.
        y_train: Training binary target labels (0 for legitimate, 1 for fraud).
        X_val: Validation feature matrix.
        X_test: Testing feature matrix.

    Returns:
        Dictionary mapping model names to a tuple (val_probs, test_probs) of
        predicted probabilities for the positive class (fraud).
    """
    y_train_arr = np.asarray(y_train)
    neg_count = np.sum(y_train_arr == 0)
    pos_count = np.sum(y_train_arr == 1)
    scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

    # 1. Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_probs_val = lr.predict_proba(X_val)[:, 1]
    lr_probs_test = lr.predict_proba(X_test)[:, 1]

    # 2. Random Forest
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    rf_probs_val = rf.predict_proba(X_val)[:, 1]
    rf_probs_test = rf.predict_proba(X_test)[:, 1]

    # 3. XGBoost
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_probs_val = xgb.predict_proba(X_val)[:, 1]
    xgb_probs_test = xgb.predict_proba(X_test)[:, 1]

    return {
        "Logistic Regression": (lr_probs_val, lr_probs_test),
        "Random Forest": (rf_probs_val, rf_probs_test),
        "XGBoost": (xgb_probs_val, xgb_probs_test),
    }
