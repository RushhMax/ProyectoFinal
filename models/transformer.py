"""Modelo Transformer para predicción de series temporales."""
import math
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (TF_D_MODEL, TF_NHEAD, TF_LAYERS,
                    TF_DROPOUT, TF_DIM_FF, WINDOW_SIZE, FEATURES)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    def __init__(self,
                 input_size: int = len(FEATURES),
                 d_model: int = TF_D_MODEL,
                 nhead: int = TF_NHEAD,
                 num_layers: int = TF_LAYERS,
                 dim_feedforward: int = TF_DIM_FF,
                 dropout: float = TF_DROPOUT,
                 seq_len: int = WINDOW_SIZE,
                 output_size: int = 1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, V)
        x = self.input_proj(x)     # (batch, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)        # (batch, T, d_model)
        x = x[:, -1]               # último token como representación
        return self.fc(x).squeeze(-1)


if __name__ == "__main__":
    model = TransformerModel()
    dummy = torch.randn(8, 20, 5)
    print(model(dummy).shape)  # (8,)
