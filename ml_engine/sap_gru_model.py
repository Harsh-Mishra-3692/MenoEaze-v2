import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# =========================================================
# LoRA Adapter (Improved)
# =========================================================
class LoRAAdapter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int = 4):
        super().__init__()

        if rank <= 0:
            raise ValueError("rank must be > 0")

        self.A = nn.Linear(hidden_dim, rank, bias=False)
        self.B = nn.Linear(rank, hidden_dim, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=0.01)
        nn.init.zeros_(self.B.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(h).all():
            return h
        return h + self.B(self.A(h))


# =========================================================
# Temporal Attention Pooling (NEW)
# =========================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h, mask=None):
        scores = self.attn(h).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)

        return pooled


# =========================================================
# SAP-GRU (Upgraded)
# =========================================================
class SAP_GRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        adapter_rank: int = 4,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.device = device or torch.device("cpu")

        # Backbone
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # NEW: attention pooling
        self.attn_pool = TemporalAttention(hidden_dim)

        # Normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Heads
        self.global_head = nn.Linear(hidden_dim, 1)
        self.local_head = nn.Linear(hidden_dim, 1)

        # Adapter
        self.adapter = LoRAAdapter(hidden_dim, adapter_rank)

        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)

        self._freeze_global()
        self.to(self.device)

    # =========================================================
    def _freeze_global(self):
        for p in self.gru.parameters():
            p.requires_grad = False
        for p in self.global_head.parameters():
            p.requires_grad = False

    # =========================================================
    def _validate_input(self, x):
        if x is None or not torch.is_tensor(x):
            raise ValueError("Invalid input")

        if x.ndim != 3:
            raise ValueError("Expected (B,T,F)")

        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("Empty input")

        if not torch.isfinite(x).all():
            raise ValueError("NaN/Inf detected")

        return torch.clamp(x, -1e6, 1e6)

    # =========================================================
    def forward(self, x, lengths=None) -> Dict[str, torch.Tensor]:

        x = self._validate_input(x)
        x = x.to(self.device, dtype=torch.float32)

        try:
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                h, _ = self.gru(packed)
                h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
            else:
                h, _ = self.gru(x)

        except Exception:
            h, _ = self.gru(x)

        # =========================
        # NEW: Attention pooling
        # =========================
        mask = None
        if lengths is not None:
            max_len = h.size(1)
            mask = torch.arange(max_len)[None, :] < lengths[:, None]

        h_pool = self.attn_pool(h, mask)

        # Normalize
        h_pool = self.norm(h_pool)

        h_pool = self.dropout(h_pool)

        # Adapter
        h_adapted = self.adapter(h_pool)

        # Heads
        y_global = self.global_head(h_pool)
        y_local = self.local_head(h_adapted)

        y = y_global + y_local

        # =========================
        # SAFETY
        # =========================
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)

        # softer output (less compression than sigmoid)
        y = torch.sigmoid(y)

        return {
            "severity": y.squeeze(-1),
            "global": y_global.squeeze(-1),
            "personal": y_local.squeeze(-1),
        }

    # =========================================================
    @torch.no_grad()
    def safe_predict(self, x, lengths=None):
        self.eval()
        try:
            return self.forward(x, lengths)
        except Exception:
            batch = x.shape[0] if x is not None else 1
            return {
                "severity": torch.zeros(batch),
                "global": torch.zeros(batch),
                "personal": torch.zeros(batch),
            }

    # =========================================================
    def assert_integrity(self):
        for p in self.gru.parameters():
            assert not p.requires_grad
        for p in self.global_head.parameters():
            assert not p.requires_grad