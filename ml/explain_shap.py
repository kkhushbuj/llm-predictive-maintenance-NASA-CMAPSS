"""Per-engine SHAP attributions: which sensors drove each RUL prediction."""
import json
import pickle

import numpy as np
import shap
import torch

from ml.config import CHECKPOINTS_DIR, FEATURE_COLUMNS, RESULTS_DIR
from ml.dataset import build_test_sequences, build_train_sequences
from ml.model import RULLSTM
from ml.preprocessing import load_dataset


def load_model(device):
    model = RULLSTM(n_features=len(FEATURE_COLUMNS)).to(device)
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / "rul_lstm_best.pt", map_location=device))
    model.eval()
    return model


class _UnsqueezedOutput(torch.nn.Module):
    """shap.GradientExplainer indexes model output as outputs[:, idx], which
    requires a 2D (batch, 1) tensor; RULLSTM.forward squeezes to 1D for the
    training loss, so wrap it just for SHAP."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).unsqueeze(-1)


def shap_per_engine(model, X_train, X_test, y_test, units, feature_columns, n_background=50, top_k=5, seed=42):
    """Per-engine attribution summed over the time window. top_sensors only
    ranks columns named sensor_*; any other inputs (e.g. regime indicators)
    are still reported in all_sensor_importance and non_sensor_share.

    `seed` controls which training windows are sampled into the background
    set — GradientExplainer's expected-gradients estimate depends on this
    sample, so varying it (see ml/shap_stability.py) measures how sensitive
    the resulting attribution is to that random choice."""
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_train), size=min(n_background, len(X_train)), replace=False)
    background = torch.from_numpy(X_train[bg_idx]).float()

    explainer = shap.GradientExplainer(_UnsqueezedOutput(model), background)
    shap_values = explainer.shap_values(torch.from_numpy(X_test).float())
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values)
    if shap_values.ndim == 4:
        shap_values = shap_values[..., 0]

    sensor_idx = [j for j, c in enumerate(feature_columns) if c.startswith("sensor_")]
    results = []
    for i, unit in enumerate(units):
        importance = np.abs(shap_values[i]).sum(axis=0)
        direction = shap_values[i].sum(axis=0)
        order = [j for j in np.argsort(-importance) if j in sensor_idx]
        top = [
            {
                "sensor": feature_columns[j],
                "importance": float(importance[j]),
                "direction": "increases_risk" if direction[j] < 0 else "decreases_risk",
                "signed_contribution": float(direction[j]),
            }
            for j in order[:top_k]
        ]
        total = float(importance.sum()) or 1.0
        results.append(
            {
                "unit": int(unit),
                "true_RUL": float(y_test[i]),
                "top_sensors": top,
                "all_sensor_importance": {feature_columns[j]: float(importance[j]) for j in range(len(feature_columns))},
                "non_sensor_share": float(
                    sum(importance[j] for j in range(len(feature_columns)) if j not in sensor_idx) / total
                ),
            }
        )
    return results


def explain(n_background: int = 50, top_k: int = 5):
    device = torch.device("cpu")  # shap gradient hooks are simplest/most stable on CPU
    model = load_model(device)

    train_df, test_df, rul_test, _ = load_dataset()
    X_train, _, _ = build_train_sequences(train_df)
    X_test, y_test, test_units = build_test_sequences(test_df)

    results = shap_per_engine(model, X_train, X_test, y_test, test_units, FEATURE_COLUMNS, n_background, top_k)

    with open(RESULTS_DIR / "shap_explanations.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved SHAP explanations for {len(results)} engines to {RESULTS_DIR / 'shap_explanations.json'}")
    return results


if __name__ == "__main__":
    explain()
