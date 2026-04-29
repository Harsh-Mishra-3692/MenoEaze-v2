"""
model_def.py
============
Single source of truth for the GRU model architecture.

This is identical to the model in train_gru.py but exists
as a standalone importable module so that api.py, maml_utils.py,
adaptation.py, and meta_train.py all use the same definition.

DO NOT modify the architecture here without also updating
train_gru.py — the two MUST match for checkpoint compatibility.
"""

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    GRU-based severity predictor.

    Architecture:
        - 2-layer GRU (hidden=64, dropout=0.2)
        - Dropout(0.2) on final hidden state
        - Linear(64 → 1) output

    Input:  (batch, seq_len=5, features=11)
    Output: (batch,) — severity score
    """

    def __init__(self, input_size: int, hidden: int = 64):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)
