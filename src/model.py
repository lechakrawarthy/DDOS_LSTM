"""
src/model.py

LSTM-based classifier for DDoS attack detection.

Architecture overview
---------------------
Input  : (batch, seq_len, n_features)
         ↓
[Optional] Input projection  (Linear → LayerNorm → GELU)
         ↓
BiLSTM  (num_layers stacked, with dropout between layers)
         ↓
Temporal Attention  (soft attention over time steps)
         ↓
Fully-connected head  (Linear → BN → GELU → Dropout → Linear)
         ↓
Output logits  (n_classes)
"""

import config as cfg
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Temporal Attention ───────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Soft attention over the LSTM output sequence.

    Computes a weighted sum  z = Σ_t  α_t · h_t
    where  α_t = softmax( w^T tanh(W h_t + b) ).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1,          bias=False)

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        lstm_out : (batch, seq_len, hidden_dim)

        Returns
        -------
        context  : (batch, hidden_dim)
        weights  : (batch, seq_len)  – for interpretability
        """
        energy = torch.tanh(self.attn(lstm_out))   # (B, T, H)
        scores = self.v(energy).squeeze(-1)         # (B, T)
        weights = F.softmax(scores, dim=1)           # (B, T)
        context = torch.bmm(weights.unsqueeze(
            1), lstm_out).squeeze(1)  # (B, H)
        return context, weights


# ─── Main Model ───────────────────────────────────────────────────────────────

class DDoSLSTM(nn.Module):
    """
    Bidirectional stacked LSTM with temporal attention for DDoS detection.

    Parameters
    ----------
    n_features   : number of input features per timestep
    n_classes    : 2 for binary classification, >2 for multi-class
    hidden_size  : LSTM hidden units (per direction)
    num_layers   : number of stacked LSTM layers
    dropout      : dropout probability (applied between LSTM layers & in MLP)
    bidirectional: whether to use BiLSTM
    proj_size    : if > 0, add an input projection layer of this size
    """

    def __init__(
        self,
        n_features:    int,
        n_classes:     int = 2,
        hidden_size:   int = cfg.HIDDEN_SIZE,
        num_layers:    int = cfg.NUM_LAYERS,
        dropout:       float = cfg.DROPOUT,
        bidirectional: bool = cfg.BIDIRECTIONAL,
        proj_size:     int = 0,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.n_dirs = 2 if bidirectional else 1
        self.hidden_size = hidden_size

        # ── Optional input projection ──────────────────────────────────────
        if proj_size > 0:
            self.input_proj = nn.Sequential(
                nn.Linear(n_features, proj_size),
                nn.LayerNorm(proj_size),
                nn.GELU(),
            )
            lstm_input = proj_size
        else:
            self.input_proj = nn.Identity()
            lstm_input = n_features

        # ── LSTM stack ─────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_size * self.n_dirs

        # ── Temporal attention ─────────────────────────────────────────────
        self.attention = TemporalAttention(lstm_out_dim)

        # ── Layer normalisation on attention output ────────────────────────
        self.layer_norm = nn.LayerNorm(lstm_out_dim)

        # ── Classification head ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.BatchNorm1d(lstm_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, n_classes),
        )

        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────────────────────
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1 (helps vanishing gradient)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        logits : (batch, n_classes)
        """
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)                 # (B, T, H*dirs)
        context, _ = self.attention(lstm_out)     # (B, H*dirs)
        context = self.layer_norm(context)
        logits = self.classifier(context)     # (B, n_classes)
        return logits

    def forward_with_attention(self, x: torch.Tensor):
        """Same as forward but also returns attention weights (B, T)."""
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        context, weights = self.attention(lstm_out)
        context = self.layer_norm(context)
        logits = self.classifier(context)
        return logits, weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_model(n_features: int, n_classes: int = 2) -> DDoSLSTM:
    model = DDoSLSTM(
        n_features=n_features,
        n_classes=n_classes,
    )
    total = model.count_parameters()
    print(f"Model built  |  Parameters: {total:,}  |  Classes: {n_classes}")
    return model


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    batch, seq, feats = 32, cfg.SEQUENCE_LEN, cfg.INPUT_SIZE
    dummy = torch.randn(batch, seq, feats)
    m = build_model(feats, n_classes=2)
    print(m)
    out = m(dummy)
    print(f"Output shape: {out.shape}")   # expected: (32, 2)
