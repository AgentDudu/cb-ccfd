from typing import Tuple
import pandas as pd


def load_and_split_data(
    file_path: str = "data/processed/creditcard_clean.parquet",
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
]:
    """Loads cleaned transaction data and performs a time-based train/val/test split.

    Args:
        file_path: Path to the cleaned parquet file.
            Defaults to "data/processed/creditcard_clean.parquet".
        test_size: Proportion of the dataset to include in the test split.
            Defaults to 0.2.
        val_size: Proportion of the dataset to include in the validation split.
            Defaults to 0.1.

    Returns:
        Tuple containing:
            - X_train: Training feature set.
            - y_train: Training labels.
            - X_val: Validation feature set.
            - y_val: Validation labels.
            - X_test: Testing feature set.
            - y_test: Testing labels.
    """
    df = pd.read_parquet(file_path)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    n_rows = len(df)
    test_start = int(n_rows * (1 - test_size))
    val_start = test_start - int(n_rows * val_size)

    X_train = X.iloc[:val_start]
    y_train = y.iloc[:val_start]
    X_val = X.iloc[val_start:test_start]
    y_val = y.iloc[val_start:test_start]
    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]

    return X_train, y_train, X_val, y_val, X_test, y_test
