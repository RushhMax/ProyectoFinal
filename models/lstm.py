"""Modelo LSTM con mecanismo de atención para predicción de series temporales."""
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT, FEATURES


class LSTMModel(nn.Module):
    def __init__(self,
                 input_size: int = len(FEATURES),
                 hidden_size: int = LSTM_HIDDEN,
                 num_layers: int = LSTM_LAYERS,
                 dropout: float = LSTM_DROPOUT,
                 output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Atención: aprende a ponderar cada paso de tiempo
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, V)
        out, _ = self.lstm(x)                              # (batch, T, hidden)
        attn_w = torch.softmax(self.attention(out), dim=1) # (batch, T, 1)
        context = (attn_w * out).sum(dim=1)                # (batch, hidden)
        return self.fc(self.dropout(context)).squeeze(-1)


if __name__ == "__main__":
    model = LSTMModel()
    dummy = torch.randn(8, 20, len(FEATURES))
    print(model(dummy).shape)  # (8,)
