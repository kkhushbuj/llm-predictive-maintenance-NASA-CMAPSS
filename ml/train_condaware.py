"""Train a Condition-Aware LSTM (embedding-based) on FD002 / FD004."""
import json
import random
import sys

import numpy as np
import shap
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.config import CHECKPOINTS_DIR, RANDOM_SEED, RESULTS_DIR, RUL_CAP, SEQUENCE_LENGTH
from ml.dataset_condaware import build_test_sequences, build_train_sequences
from ml.metrics import nasa_score, rmse
from ml.model_condaware import ConditionAwareLSTM
from ml.multicond import load_multicond, REGIME_COLUMNS

HPARAMS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "lr": 1e-3,
    "batch_size": 256,
    "max_epochs": 100,
    "patience": 15,
    "val_engine_fraction": 0.15,
    "sequence_length": SEQUENCE_LENGTH,
    "rul_cap": RUL_CAP,
    "embed_dim": 4,
    "n_regimes": 6,
}
TOP_RISK_N = 15


def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


class _SensorOnlyWrapper(torch.nn.Module):
    """Wraps ConditionAwareLSTM so SHAP sees a single-tensor model.
    Regime IDs are frozen at construction time."""
    def __init__(self, model, regime_ids):
        super().__init__()
        self.model = model
        self.register_buffer("regime_ids", regime_ids)

    def forward(self, sensors):
        batch = sensors.shape[0]
        rids = self.regime_ids.expand(batch, -1)
        return self.model(sensors, rids).unsqueeze(-1)


def shap_per_engine(model, S_train, R_train, S_test, y_test, units, sensor_cols,
                    n_background=50, top_k=5):
    rng = np.random.default_rng(RANDOM_SEED)
    bg_idx = rng.choice(len(S_train), size=min(n_background, len(S_train)), replace=False)
    bg_sensors = torch.from_numpy(S_train[bg_idx]).float()
    bg_regimes = torch.from_numpy(R_train[bg_idx]).long()

    regime_mode = torch.mode(bg_regimes, dim=0).values
    wrapper = _SensorOnlyWrapper(model, regime_mode.unsqueeze(0))
    wrapper.eval()

    explainer = shap.GradientExplainer(wrapper, bg_sensors)
    sv = explainer.shap_values(torch.from_numpy(S_test).float())
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.array(sv)
    if sv.ndim == 4:
        sv = sv[..., 0]

    sensor_idx = list(range(len(sensor_cols)))
    results = []
    for i, unit in enumerate(units):
        importance = np.abs(sv[i]).sum(axis=0)
        direction = sv[i].sum(axis=0)
        order = np.argsort(-importance).tolist()
        top = [
            {
                "sensor": sensor_cols[j],
                "importance": float(importance[j]),
                "direction": "increases_risk" if direction[j] < 0 else "decreases_risk",
                "signed_contribution": float(direction[j]),
            }
            for j in order[:top_k]
        ]
        total = float(importance.sum()) or 1.0
        results.append({
            "unit": int(unit),
            "true_RUL": float(y_test[i]),
            "top_sensors": top,
            "all_sensor_importance": {sensor_cols[j]: float(importance[j]) for j in sensor_idx},
            "non_sensor_share": 0.0,
        })
    return results


def run(subset: str):
    set_seed()
    tag = f"{subset}_condaware"
    out_dir = RESULTS_DIR / tag
    ckpt_dir = CHECKPOINTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, features, audit, regime_info = load_multicond(subset)
    sensor_cols = [c for c in features if c not in REGIME_COLUMNS]

    with open(out_dir / "sensor_audit.json", "w") as f:
        json.dump({"features_used": sensor_cols, "embed_dim": HPARAMS["embed_dim"],
                    "sensors": audit, "regimes": regime_info}, f, indent=2)

    rng = np.random.default_rng(RANDOM_SEED)
    engines = train_df["unit"].unique()
    val_engines = set(rng.choice(engines, size=int(len(engines) * HPARAMS["val_engine_fraction"]), replace=False))
    tr_part = train_df[~train_df["unit"].isin(val_engines)]
    va_part = train_df[train_df["unit"].isin(val_engines)]

    S_tr, R_tr, y_tr, _ = build_train_sequences(tr_part, sensor_cols)
    S_va, R_va, y_va, _ = build_train_sequences(va_part, sensor_cols)
    S_te, R_te, y_te, te_units = build_test_sequences(test_df, sensor_cols)

    def make_loader(S, R, y, shuffle=False):
        ds = TensorDataset(torch.from_numpy(S), torch.from_numpy(R),
                           torch.from_numpy(y / RUL_CAP))
        return DataLoader(ds, batch_size=HPARAMS["batch_size"], shuffle=shuffle)

    train_loader = make_loader(S_tr, R_tr, y_tr, shuffle=True)
    val_loader = make_loader(S_va, R_va, y_va)

    model = ConditionAwareLSTM(
        n_sensors=len(sensor_cols),
        n_regimes=HPARAMS["n_regimes"],
        embed_dim=HPARAMS["embed_dim"],
        hidden_size=HPARAMS["hidden_size"],
        num_layers=HPARAMS["num_layers"],
        dropout=HPARAMS["dropout"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"])
    loss_fn = torch.nn.MSELoss()

    best_val, stale, history = float("inf"), 0, []
    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        model.train()
        tl = []
        for sb, rb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(sb, rb), yb)
            loss.backward()
            optimizer.step()
            tl.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = [loss_fn(model(sb, rb), yb).item() for sb, rb, yb in val_loader]
        train_rmse = float(np.sqrt(np.mean(tl))) * RUL_CAP
        val_rmse = float(np.sqrt(np.mean(vl))) * RUL_CAP
        history.append({"epoch": epoch, "train_rmse_approx": train_rmse, "val_rmse_approx": val_rmse})
        print(f"[{tag}] epoch {epoch:3d}  train~{train_rmse:6.2f}  val~{val_rmse:6.2f}", flush=True)
        if val_rmse < best_val - 1e-3:
            best_val, stale = val_rmse, 0
            torch.save(model.state_dict(), ckpt_dir / "rul_lstm_best.pt")
        else:
            stale += 1
            if stale >= HPARAMS["patience"]:
                print(f"[{tag}] early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(torch.load(ckpt_dir / "rul_lstm_best.pt"))
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(S_te), torch.from_numpy(R_te)).numpy() * RUL_CAP

    metrics = {
        "subset": subset,
        "model": "ConditionAwareLSTM",
        "n_train_engines": int(len(engines) - len(val_engines)),
        "n_val_engines": int(len(val_engines)),
        "n_test_engines": int(len(te_units)),
        "sensor_features": sensor_cols,
        "hparams": HPARAMS,
        "best_val_rmse": best_val,
        "test_rmse": rmse(y_te, pred),
        "test_nasa_score": nasa_score(y_te, pred),
        "test_nasa_score_per_engine": nasa_score(y_te, pred) / len(te_units),
        "history": history,
    }
    with open(out_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{tag}] TEST RMSE {metrics['test_rmse']:.3f}   NASA {metrics['test_nasa_score']:.1f}", flush=True)

    preds = [
        {"unit": int(u), "true_RUL": float(t), "predicted_RUL": float(max(0.0, p)), "error": float(p - t)}
        for u, t, p in zip(te_units, y_te, pred)
    ]
    max_pred = max(p["predicted_RUL"] for p in preds) or 1.0
    for p in preds:
        p["risk_score"] = round(100 * (1 - p["predicted_RUL"] / max_pred), 2)
    preds.sort(key=lambda p: p["predicted_RUL"])
    for rank, p in enumerate(preds, start=1):
        p["risk_rank"] = rank
        p["top15_risk"] = rank <= TOP_RISK_N
    with open(out_dir / "risk_scores.json", "w") as f:
        json.dump(preds, f, indent=2)

    shap_results = shap_per_engine(model, S_tr, R_tr, S_te, y_te, te_units, sensor_cols)
    with open(out_dir / "shap_explanations.json", "w") as f:
        json.dump(shap_results, f, indent=2)
    print(f"[{tag}] saved SHAP for {len(shap_results)} engines", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:] or ["FD002", "FD004"]:
        run(s)
