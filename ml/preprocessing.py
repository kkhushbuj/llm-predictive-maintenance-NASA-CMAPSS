"""Load, label, and normalize the NASA C-MAPSS FD001 dataset."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ml.config import COLUMNS, DATA_DIR, FEATURE_COLUMNS, RUL_CAP, SUBSET


def _read_txt(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    return df


def load_raw(subset: str = SUBSET):
    train = _read_txt(DATA_DIR / f"train_{subset}.txt")
    test = _read_txt(DATA_DIR / f"test_{subset}.txt")
    rul_test = pd.read_csv(DATA_DIR / f"RUL_{subset}.txt", header=None, names=["RUL"])
    return train, test, rul_test


def add_rul_train(train: pd.DataFrame) -> pd.DataFrame:
    """Each unit fails at its last recorded cycle; RUL counts down to 0 there."""
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train = train.copy()
    train["RUL"] = max_cycle - train["cycle"]
    train["RUL"] = train["RUL"].clip(upper=RUL_CAP)
    return train


def add_rul_test(test: pd.DataFrame, rul_test: pd.DataFrame) -> pd.DataFrame:
    """Test trajectories are truncated before failure; true final RUL is given
    separately, so we back-compute RUL at every row from that anchor."""
    test = test.copy()
    final_rul = rul_test["RUL"].values
    max_cycle = test.groupby("unit")["cycle"].transform("max")
    unit_to_final = {u: final_rul[i] for i, u in enumerate(sorted(test["unit"].unique()))}
    test["final_RUL"] = test["unit"].map(unit_to_final)
    test["RUL"] = (max_cycle - test["cycle"]) + test["final_RUL"]
    test["RUL"] = test["RUL"].clip(upper=RUL_CAP)
    test = test.drop(columns=["final_RUL"])
    return test


def fit_scaler(train: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(train[FEATURE_COLUMNS])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    df = df.copy()
    df[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])
    return df


def load_dataset(subset: str = SUBSET):
    """Returns (train_df, test_df, rul_test_df, scaler) fully labeled and scaled."""
    train, test, rul_test = load_raw(subset)
    train = add_rul_train(train)
    test_labeled = add_rul_test(test, rul_test)

    scaler = fit_scaler(train)
    train_scaled = apply_scaler(train, scaler)
    test_scaled = apply_scaler(test_labeled, scaler)

    return train_scaled, test_scaled, rul_test, scaler


if __name__ == "__main__":
    train, test, rul_test, scaler = load_dataset()
    print("Train shape:", train.shape, "engines:", train["unit"].nunique())
    print("Test shape:", test.shape, "engines:", test["unit"].nunique())
    print(train[["unit", "cycle", "RUL"]].head())
