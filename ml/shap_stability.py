"""SHAP-stability audit: how much does an engine's top-attributed sensor
change if GradientExplainer's random background sample is resampled?

The Critique Agent's standing hypothesis for a "Novel"/"Contradicted" verdict
is often "maybe this is just SHAP background-sampling noise" — but until now
it had no data to check that hypothesis against, only its own judgment. This
module gives it one: rerun `shap_per_engine` several times with different
background samples (same trained model, same test point) and measure, per
engine, how consistent the #1 sensor and its importance are across runs.

Requires a trained checkpoint (`ml/train_lstm.py` for FD001,
`ml/train_multicond.py <SUBSET>` for FD002/FD004) — it explains an existing
model, it does not train one.

    python -m ml.shap_stability FD001
    python -m ml.shap_stability FD002 FD004
"""
import json
import sys
from collections import Counter

import numpy as np
import torch

from ml.config import CHECKPOINTS_DIR, FEATURE_COLUMNS, RESULTS_DIR
from ml.dataset import build_test_sequences, build_train_sequences
from ml.explain_shap import _UnsqueezedOutput, shap_per_engine
from ml.model import RULLSTM
from ml.multicond import load_multicond
from ml.preprocessing import load_dataset

N_RUNS = 5
SEEDS = [42, 43, 44, 45, 46]  # first matches explain_shap.py's default, so run 0 is directly comparable
AGREEMENT_THRESHOLD = 0.8  # >= this fraction of runs must agree on the #1 sensor to call it "stable"
CV_THRESHOLD = 0.35  # coefficient of variation of that sensor's importance, above which we call it "noisy"


def _load_model(n_features, ckpt_dir):
    model = RULLSTM(n_features=n_features)
    model.load_state_dict(torch.load(ckpt_dir / "rul_lstm_best.pt", map_location="cpu"))
    model.eval()
    return model


def compute_stability(model, X_train, X_test, y_test, units, feature_columns, n_background=50, top_k=5):
    """Runs shap_per_engine once per seed in SEEDS and reduces the results to
    one stability record per engine."""
    runs = [
        shap_per_engine(model, X_train, X_test, y_test, units, feature_columns, n_background, top_k, seed=seed)
        for seed in SEEDS
    ]
    return reduce_runs(runs, units)


def reduce_runs(runs: list[list[dict]], units) -> list[dict]:
    """Pure reduction step, split out from compute_stability so it can be unit
    tested with synthetic per-engine SHAP results instead of a real model."""
    stability = []
    for i, unit in enumerate(units):
        top_sensor_per_run = [r[i]["top_sensors"][0]["sensor"] for r in runs if r[i]["top_sensors"]]
        if not top_sensor_per_run:
            continue
        mode_sensor, mode_count = Counter(top_sensor_per_run).most_common(1)[0]
        agreement_rate = mode_count / len(top_sensor_per_run)

        mode_importances = [
            r[i]["all_sensor_importance"][mode_sensor] for r in runs if mode_sensor in r[i]["all_sensor_importance"]
        ]
        mean_importance = float(np.mean(mode_importances))
        std_importance = float(np.std(mode_importances))
        cv_importance = std_importance / mean_importance if mean_importance > 1e-9 else 0.0

        stability.append(
            {
                "unit": int(unit),
                "n_runs": len(top_sensor_per_run),
                "top_sensor_per_run": top_sensor_per_run,
                "mode_top_sensor": mode_sensor,
                "top_sensor_agreement_rate": round(agreement_rate, 3),
                "mode_sensor_importance_mean": mean_importance,
                "mode_sensor_importance_cv": round(cv_importance, 3),
                "stable": bool(agreement_rate >= AGREEMENT_THRESHOLD and cv_importance <= CV_THRESHOLD),
            }
        )
    return stability


def explain_stability(subset: str = "FD001", n_background: int = 50, top_k: int = 5):
    ckpt_dir = CHECKPOINTS_DIR if subset == "FD001" else CHECKPOINTS_DIR / subset
    out_dir = RESULTS_DIR if subset == "FD001" else RESULTS_DIR / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    if subset == "FD001":
        train_df, test_df, _, _ = load_dataset()
        feature_columns = FEATURE_COLUMNS
    else:
        train_df, test_df, feature_columns, _, _ = load_multicond(subset)

    X_train, _, _ = build_train_sequences(train_df, feature_columns=feature_columns)
    X_test, y_test, units = build_test_sequences(test_df, feature_columns=feature_columns)

    model = _load_model(len(feature_columns), ckpt_dir)
    stability = compute_stability(model, X_train, X_test, y_test, units, feature_columns, n_background, top_k)

    out_path = out_dir / "shap_stability.json"
    with open(out_path, "w") as f:
        json.dump(stability, f, indent=2)

    n_stable = sum(1 for s in stability if s["stable"])
    print(f"[{subset}] {n_stable}/{len(stability)} engines have a stable top sensor -> {out_path}")
    return stability


if __name__ == "__main__":
    for s in sys.argv[1:] or ["FD001"]:
        explain_stability(s)
