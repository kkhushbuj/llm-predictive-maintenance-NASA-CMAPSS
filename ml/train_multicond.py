"""Train, evaluate and explain an LSTM on FD002 or FD004, from scratch.

Why not transfer learning from the FD001 model: see REPORT_FD002_FD004.md
(short version: different input space, target data is larger than source,
FD004's fan fault never appears in FD001, and published per-subset
benchmarks are trained on that subset alone).

All settings below were fixed before any FD002/FD004 result was produced.
Differences from the FD001 run are deliberate and listed in the report:
- engine-level validation split (FD001 split windows at random, so windows
  from the same engine could sit in both train and validation);
- target scaled by RUL_CAP (FD001 training sat on a flat plateau for ~20
  epochs predicting the mean before learning; scaling avoids that);
- patience 15 / max 100 epochs to match the larger datasets.
"""
import json
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.config import CHECKPOINTS_DIR, RANDOM_SEED, RESULTS_DIR, RUL_CAP, SEQUENCE_LENGTH
from ml.dataset import build_test_sequences, build_train_sequences
from ml.explain_shap import shap_per_engine
from ml.metrics import nasa_score, rmse
from ml.model import RULLSTM
from ml.multicond import load_multicond

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
}
TOP_RISK_N = 15


def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def run(subset: str):
    set_seed()
    out_dir = RESULTS_DIR / subset
    ckpt_dir = CHECKPOINTS_DIR / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, features, audit, regime_info = load_multicond(subset)
    with open(out_dir / "sensor_audit.json", "w") as f:
        json.dump({"features_used": features, "sensors": audit, "regimes": regime_info}, f, indent=2)

    rng = np.random.default_rng(RANDOM_SEED)
    engines = train_df["unit"].unique()
    val_engines = set(rng.choice(engines, size=int(len(engines) * HPARAMS["val_engine_fraction"]), replace=False))
    tr_part = train_df[~train_df["unit"].isin(val_engines)]
    va_part = train_df[train_df["unit"].isin(val_engines)]

    X_tr, y_tr, _ = build_train_sequences(tr_part, feature_columns=features)
    X_va, y_va, _ = build_train_sequences(va_part, feature_columns=features)
    X_te, y_te, te_units = build_test_sequences(test_df, feature_columns=features)

    to_ds = lambda X, y: TensorDataset(torch.from_numpy(X), torch.from_numpy(y / RUL_CAP))
    train_loader = DataLoader(to_ds(X_tr, y_tr), batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader = DataLoader(to_ds(X_va, y_va), batch_size=HPARAMS["batch_size"])

    model = RULLSTM(len(features), HPARAMS["hidden_size"], HPARAMS["num_layers"], HPARAMS["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"])
    loss_fn = torch.nn.MSELoss()

    best_val, stale, history = float("inf"), 0, []
    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        model.train()
        tl = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            tl.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = [loss_fn(model(xb), yb).item() for xb, yb in val_loader]
        train_rmse = float(np.sqrt(np.mean(tl))) * RUL_CAP
        val_rmse = float(np.sqrt(np.mean(vl))) * RUL_CAP
        history.append({"epoch": epoch, "train_rmse_approx": train_rmse, "val_rmse_approx": val_rmse})
        print(f"[{subset}] epoch {epoch:3d}  train_rmse~{train_rmse:6.2f}  val_rmse~{val_rmse:6.2f}", flush=True)
        if val_rmse < best_val - 1e-3:
            best_val, stale = val_rmse, 0
            torch.save(model.state_dict(), ckpt_dir / "rul_lstm_best.pt")
        else:
            stale += 1
            if stale >= HPARAMS["patience"]:
                print(f"[{subset}] early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(torch.load(ckpt_dir / "rul_lstm_best.pt"))
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_te)).numpy() * RUL_CAP

    metrics = {
        "subset": subset,
        "n_train_engines": int(len(engines) - len(val_engines)),
        "n_val_engines": int(len(val_engines)),
        "n_test_engines": int(len(te_units)),
        "features_used": features,
        "hparams": HPARAMS,
        "best_val_rmse": best_val,
        "test_rmse": rmse(y_te, pred),
        "test_nasa_score": nasa_score(y_te, pred),
        "test_nasa_score_per_engine": nasa_score(y_te, pred) / len(te_units),
        "history": history,
    }
    with open(out_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{subset}] TEST RMSE {metrics['test_rmse']:.3f}   NASA score {metrics['test_nasa_score']:.1f}", flush=True)

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

    shap_results = shap_per_engine(model, X_tr, X_te, y_te, te_units, features)
    with open(out_dir / "shap_explanations.json", "w") as f:
        json.dump(shap_results, f, indent=2)
    print(f"[{subset}] saved SHAP for {len(shap_results)} engines", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:] or ["FD002", "FD004"]:
        run(s)
