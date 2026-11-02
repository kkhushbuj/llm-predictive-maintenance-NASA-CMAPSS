import json
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from ml.config import (
    CHECKPOINTS_DIR,
    FEATURE_COLUMNS,
    PROCESSED_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SEQUENCE_LENGTH,
)
from ml.dataset import RULDataset, build_test_sequences, build_train_sequences
from ml.metrics import nasa_score, rmse
from ml.model import RULLSTM
from ml.preprocessing import load_dataset


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(epochs=60, batch_size=256, lr=1e-3, patience=8):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, test_df, rul_test, scaler = load_dataset()
    X_train, y_train, _ = build_train_sequences(train_df)
    X_test, y_test, test_units = build_test_sequences(test_df)

    full_ds = RULDataset(X_train, y_train)
    n_val = int(0.15 * len(full_ds))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = RULLSTM(n_features=len(FEATURE_COLUMNS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:3d}  train_mse {train_loss:8.3f}  val_mse {val_loss:8.3f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "rul_lstm_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(CHECKPOINTS_DIR / "rul_lstm_best.pt"))
    model.eval()

    with torch.no_grad():
        test_pred = model(torch.from_numpy(X_test).to(device)).cpu().numpy()

    test_rmse = rmse(y_test, test_pred)
    test_score = nasa_score(y_test, test_pred)
    print(f"\nTest RMSE: {test_rmse:.3f}   NASA Score: {test_score:.1f}")

    with open(CHECKPOINTS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    metrics = {
        "subset": "FD001",
        "sequence_length": SEQUENCE_LENGTH,
        "feature_columns": FEATURE_COLUMNS,
        "test_rmse": test_rmse,
        "test_nasa_score": test_score,
        "n_train_engines": int(train_df["unit"].nunique()),
        "n_test_engines": int(test_df["unit"].nunique()),
        "history": history,
    }
    with open(RESULTS_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    per_engine = []
    for u, true_r, pred_r in zip(test_units, y_test, test_pred):
        per_engine.append(
            {
                "unit": int(u),
                "true_RUL": float(true_r),
                "predicted_RUL": float(max(0.0, pred_r)),
                "error": float(pred_r - true_r),
            }
        )
    with open(RESULTS_DIR / "test_predictions.json", "w") as f:
        json.dump(per_engine, f, indent=2)

    train_df.to_parquet(PROCESSED_DIR / "train_FD001.parquet")
    test_df.to_parquet(PROCESSED_DIR / "test_FD001.parquet")

    print(f"\nSaved checkpoint, scaler, metrics, and predictions to {RESULTS_DIR}")
    return metrics


if __name__ == "__main__":
    train()
