"""Sliding-window sequence construction for the LSTM."""
import numpy as np
import torch
from torch.utils.data import Dataset

from ml.config import FEATURE_COLUMNS, SEQUENCE_LENGTH


def build_train_sequences(train_df, seq_len: int = SEQUENCE_LENGTH, feature_columns=FEATURE_COLUMNS):
    """One sample per (engine, window-end-cycle) pair, using all available
    history — short trajectories are left-padded by repeating the first row."""
    sequences, targets, units = [], [], []
    for unit, g in train_df.groupby("unit"):
        g = g.sort_values("cycle")
        feats = g[feature_columns].values
        ruls = g["RUL"].values
        n = len(g)
        for end in range(1, n + 1):
            start = max(0, end - seq_len)
            window = feats[start:end]
            if len(window) < seq_len:
                pad = np.repeat(window[:1], seq_len - len(window), axis=0)
                window = np.concatenate([pad, window], axis=0)
            sequences.append(window)
            targets.append(ruls[end - 1])
            units.append(unit)
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(units)


def build_test_sequences(test_df, seq_len: int = SEQUENCE_LENGTH, feature_columns=FEATURE_COLUMNS):
    """One sample per engine: the last `seq_len` cycles observed, used to
    predict RUL at the point the trajectory was truncated."""
    sequences, targets, units = [], [], []
    for unit, g in test_df.groupby("unit"):
        g = g.sort_values("cycle")
        feats = g[feature_columns].values
        window = feats[-seq_len:]
        if len(window) < seq_len:
            pad = np.repeat(window[:1], seq_len - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)
        sequences.append(window)
        targets.append(g["RUL"].values[-1])
        units.append(unit)
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(units)


class RULDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.X = torch.from_numpy(sequences)
        self.y = torch.from_numpy(targets)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
