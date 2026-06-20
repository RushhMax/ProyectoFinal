"""
Entrenamiento de los tres modelos heterogéneos:
  LSTM, Transformer (PyTorch) y Random Forest (sklearn).

Uso: python train.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys, json

sys.path.insert(0, str(Path(__file__).parent))
from config import (BATCH_SIZE, EPOCHS, LR, PATIENCE,
                    MODELS_DIR, RESULTS_DIR, RANDOM_STATE)
from data.download import download_sp500, save_raw, load_raw
from data.preprocessing import prepare_data_with_val
from models.lstm import LSTMModel
from models.transformer import TransformerModel
from models.random_forest import RandomForestModel

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Utilidades de entrenamiento PyTorch ───────────────────────────────────────

def make_loaders(X_train, y_train, X_val=None, y_val=None):
    X_t = torch.tensor(X_train)
    y_t = torch.tensor(y_train)
    train_ds = TensorDataset(X_t, y_t)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = None
    if X_val is not None:
        val_dl = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
            batch_size=BATCH_SIZE,
        )
    return train_dl, val_dl


def train_pytorch(model: nn.Module,
                  train_dl: DataLoader,
                  val_dl: DataLoader,
                  name: str,
                  weight_decay: float = 1e-4) -> dict:
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = nn.MSELoss()
    best_val, patience_cnt = float("inf"), 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_dl.dataset)

        val_loss = 0.0
        if val_dl is not None:
            model.eval()
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    val_loss += criterion(model(xb), yb).item() * len(xb)
            val_loss /= len(val_dl.dataset)
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [{name}] Epoch {epoch:03d} | "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  lr={lr_now:.2e}")

        # Early stopping
        if val_dl is not None:
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience_cnt = 0
                torch.save(model.state_dict(), MODELS_DIR / f"{name}_best.pt")
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"  [{name}] Early stopping en epoch {epoch}")
                    break

    if val_dl is not None:
        model.load_state_dict(torch.load(MODELS_DIR / f"{name}_best.pt",
                                         weights_only=True))
    return history


def evaluate_pytorch(model: nn.Module,
                     X_test: np.ndarray,
                     y_test: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
    mse = float(np.mean((pred - y_test) ** 2))
    mae = float(np.mean(np.abs(pred - y_test)))
    return {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse))}


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    # 1. Datos
    raw_path = Path("data/sp500_raw.csv")
    if raw_path.exists():
        print("Cargando datos guardados...")
        df = load_raw()
    else:
        df = download_sp500()
        save_raw(df)

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data_with_val(df)

    metrics = {}

    # 2. LSTM
    print("\n=== Entrenando LSTM ===")
    lstm = LSTMModel()
    train_dl, val_dl = make_loaders(X_train, y_train, X_val, y_val)
    train_pytorch(lstm, train_dl, val_dl, "lstm")
    metrics["lstm"] = evaluate_pytorch(lstm, X_test, y_test)
    print(f"  LSTM test: {metrics['lstm']}")

    # 3. Transformer
    print("\n=== Entrenando Transformer ===")
    transformer = TransformerModel()
    train_pytorch(transformer, train_dl, val_dl, "transformer")
    metrics["transformer"] = evaluate_pytorch(transformer, X_test, y_test)
    print(f"  Transformer test: {metrics['transformer']}")

    # 4. Random Forest
    print("\n=== Entrenando Random Forest ===")
    rf = RandomForestModel()
    rf.fit(X_train, y_train)
    rf.save()
    pred_rf = rf.predict(X_test)
    mse = float(np.mean((pred_rf - y_test) ** 2))
    metrics["random_forest"] = {
        "mse": mse, "mae": float(np.mean(np.abs(pred_rf - y_test))),
        "rmse": float(np.sqrt(mse)),
    }
    print(f"  RF test: {metrics['random_forest']}")

    # 5. Guardar métricas y checkpoints finales
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    torch.save(lstm.state_dict(), MODELS_DIR / "lstm_final.pt")
    torch.save(transformer.state_dict(), MODELS_DIR / "transformer_final.pt")
    print("\nEntrenamiento completado. Métricas guardadas en results/metrics.json")


if __name__ == "__main__":
    main()
