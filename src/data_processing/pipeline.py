from typing import Tuple
import pandas as pd


def load_and_split_data(
    file_path: str = "data/processed/creditcard_clean.parquet",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Loads cleaned transaction data and performs a time-based train/test split.

    Args:
        file_path: Path to the cleaned parquet file.
            Defaults to "data/processed/creditcard_clean.parquet".
        test_size: Proportion of the dataset to include in the test split.
            Defaults to 0.2.

    Returns:
        Tuple containing:
            - X_train: Training feature set.
            - y_train: Training labels.
            - X_test: Testing feature set.
            - y_test: Testing labels.
    """
    df = pd.read_parquet(file_path)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    split_idx = int(len(df) * (1 - test_size))

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    return X_train, y_train, X_test, y_test
