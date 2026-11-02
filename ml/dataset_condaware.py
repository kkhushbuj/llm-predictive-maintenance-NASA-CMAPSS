"""Sequence builders that return sensor features and regime IDs as separate arrays."""
import numpy as np

from ml.config import SEQUENCE_LENGTH


def _build_sequences(df, sensor_cols, seq_len, mode):
    """Shared logic for train (all windows) and test (last window per engine).
    Returns (sensors, regime_ids, targets, units) as numpy arrays."""
    sensors, regimes, targets, units = [], [], [], []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        feats = g[sensor_cols].values.astype(np.float32)
        rids = g["regime"].values.astype(np.int64)
        ruls = g["RUL"].values.astype(np.float32)

        if mode == "train":
            n = len(g)
            for end in range(1, n + 1):
                start = max(0, end - seq_len)
                s_win = feats[start:end]
                r_win = rids[start:end]
                if len(s_win) < seq_len:
                    pad_n = seq_len - len(s_win)
                    s_win = np.concatenate([np.repeat(s_win[:1], pad_n, axis=0), s_win])
                    r_win = np.concatenate([np.repeat(r_win[:1], pad_n), r_win])
                sensors.append(s_win)
                regimes.append(r_win)
                targets.append(ruls[end - 1])
                units.append(unit)
        else:
            s_win = feats[-seq_len:]
            r_win = rids[-seq_len:]
            if len(s_win) < seq_len:
                pad_n = seq_len - len(s_win)
                s_win = np.concatenate([np.repeat(s_win[:1], pad_n, axis=0), s_win])
                r_win = np.concatenate([np.repeat(r_win[:1], pad_n), r_win])
            sensors.append(s_win)
            regimes.append(r_win)
            targets.append(ruls[-1])
            units.append(unit)

    return (np.array(sensors, dtype=np.float32),
            np.array(regimes, dtype=np.int64),
            np.array(targets, dtype=np.float32),
            np.array(units))


def build_train_sequences(df, sensor_cols, seq_len=SEQUENCE_LENGTH):
    return _build_sequences(df, sensor_cols, seq_len, "train")


def build_test_sequences(df, sensor_cols, seq_len=SEQUENCE_LENGTH):
    return _build_sequences(df, sensor_cols, seq_len, "test")
