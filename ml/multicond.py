"""Data handling for the multi-condition subsets FD002 / FD004.

Differs from the FD001 path (ml/preprocessing.py) in three ways:
- Each row's operating regime is identified with k-means (k=6) on the three
  operational settings, fit on the training set only.
- Sensors are z-scored within their regime using training-set statistics, so
  a value is measured against what is normal for that flight condition rather
  than against the whole fleet (global scaling would mostly encode regime).
- Sensor selection is re-derived from this data by audit_sensors(), not copied
  from FD001: a sensor is dropped only if it is constant inside every regime.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ml.config import CONSTANT_SENSORS as FD001_DROPPED, RANDOM_SEED
from ml.preprocessing import add_rul_test, add_rul_train, load_raw

OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
N_REGIMES = 6
Z_CLIP = 10.0
REGIME_COLUMNS = [f"regime_{k}" for k in range(N_REGIMES)]


def fit_regimes(train: pd.DataFrame) -> KMeans:
    return KMeans(n_clusters=N_REGIMES, n_init=10, random_state=RANDOM_SEED).fit(train[OP_SETTINGS])


def audit_sensors(train: pd.DataFrame) -> list[dict]:
    """Per-sensor evidence for the keep/drop decision, computed on train only."""
    rows = []
    for s in SENSORS:
        by_regime = train.groupby("regime")[s]
        within_std = by_regime.std()
        z = (train[s] - by_regime.transform("mean")) / by_regime.transform("std").replace(0, np.nan)
        rho = z.corr(train["RUL"], method="spearman") if z.notna().any() else None
        constant_in_every_regime = bool((by_regime.nunique() == 1).all())
        rows.append(
            {
                "sensor": s,
                "dropped_in_FD001": s in FD001_DROPPED,
                "global_std": float(train[s].std()),
                "max_within_regime_std": float(within_std.max()),
                "max_unique_values_in_a_regime": int(by_regime.nunique().max()),
                "avg_share_of_modal_value": float(
                    by_regime.agg(lambda x: x.value_counts(normalize=True).iloc[0]).mean()
                ),
                "within_regime_spearman_vs_RUL": None if rho is None or np.isnan(rho) else float(rho),
                "constant_in_every_regime": constant_in_every_regime,
                "kept": not constant_in_every_regime,
            }
        )
    return rows


def load_multicond(subset: str):
    """Returns (train_df, test_df, feature_columns, audit, regime_info)."""
    train, test, rul_test = load_raw(subset)
    train = add_rul_train(train)
    test = add_rul_test(test, rul_test)

    km = fit_regimes(train)
    train["regime"] = km.labels_
    test["regime"] = km.predict(test[OP_SETTINGS])

    audit = audit_sensors(train)
    kept = [row["sensor"] for row in audit if row["kept"]]

    stats = train.groupby("regime")[kept].agg(["mean", "std"])
    for df in (train, test):
        mean = stats.xs("mean", axis=1, level=1).loc[df["regime"]].to_numpy()
        std = stats.xs("std", axis=1, level=1).loc[df["regime"]].to_numpy()
        # A kept sensor can still be constant inside one particular regime.
        std = np.where((std == 0) | np.isnan(std), 1.0, std)
        # Quasi-binary sensors (e.g. 10 in FD002, 16 in FD004) have a tiny
        # in-regime std, so one flip gives z~50. Continuous sensors peak ~8.6.
        df[kept] = np.clip((df[kept].to_numpy() - mean) / std, -Z_CLIP, Z_CLIP)
        for k, col in enumerate(REGIME_COLUMNS):
            df[col] = (df["regime"] == k).astype(np.float32)

    op_dist_test = np.linalg.norm(test[OP_SETTINGS].to_numpy() - km.cluster_centers_[test["regime"]], axis=1)
    regime_info = {
        "centroids": km.cluster_centers_.round(4).tolist(),
        "train_rows_per_regime": np.bincount(train["regime"], minlength=N_REGIMES).tolist(),
        "max_test_distance_to_centroid": float(op_dist_test.max()),
    }
    return train, test, kept + REGIME_COLUMNS, audit, regime_info
