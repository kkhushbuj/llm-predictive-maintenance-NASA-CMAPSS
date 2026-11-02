"""Condition-Aware LSTM: replaces one-hot regime columns with a learned embedding."""
import torch
import torch.nn as nn


class ConditionAwareLSTM(nn.Module):
    def __init__(self, n_sensors: int, n_regimes: int = 6, embed_dim: int = 4,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.regime_embed = nn.Embedding(n_regimes, embed_dim)
        self.lstm = nn.LSTM(
            input_size=n_sensors + embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, sensors, regime_ids):
        emb = self.regime_embed(regime_ids)
        x = torch.cat([sensors, emb], dim=-1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)
