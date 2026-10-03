"""
Experimento secundario: robustez frente a autocorrelación trivial.

Repite el entrenamiento de LSTM, Transformer y Random Forest usando como
variable objetivo el retorno logarítmico del día siguiente (Log_Return)
en lugar del precio de cierre normalizado (Close). Esto rompe la
autocorrelación trivial señalada en Limitaciones: predecir "sin cambio"
(retorno=0) no coincide con reproducir el nivel de precio actual.

No modifica config.py ni los checkpoints/resultados del target original
(Close): usa nombres de archivo separados (*_logret) en saved_models/ y
guarda sus propios resultados en results/logreturn_*.

Uso: python -m experiments.exp_logreturn
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from config import (FEATURES, WINDOW_SIZE, HORIZON, TRAIN_RATIO, VAL_RATIO,
                     MODELS_DIR, RESULTS_DIR, RANDOM_STATE, N_EXPLAIN)
from data.download import load_raw
from data.features import add_indicators
from data.preprocessing import split_temporal, load_scaler, scale, make_windows
from models.lstm import LSTMModel
from models.transformer import TransformerModel
from models.random_forest import RandomForestModel
from framework.explainer import ModelAgnosticExplainer
from evaluation.consistency import mean_pairwise_consistency
from evaluation.significance import per_model_significance
from train import make_loaders, train_pytorch, evaluate_pytorch

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
DEVICE = "cpu"

TARGET_COL = "Log_Return"
target_idx = FEATURES.index(TARGET_COL)


def main():
    t0 = time.time()

    # 1. Datos — mismo split y mismo escalador que el experimento principal,
    #    solo cambia la columna objetivo al construir las ventanas.
    df = add_indicators(load_raw())
    dev_df, test_df = split_temporal(df, TRAIN_RATIO)
    scaler = load_scaler()  # ya cubre Log_Return (se ajustó sobre todas las FEATURES)

    cut = int(len(dev_df) * (1.0 - VAL_RATIO))
    train_df = dev_df.iloc[:cut]
    val_df = dev_df.iloc[cut:]

    X_train, y_train = make_windows(scale(train_df, scaler), WINDOW_SIZE, HORIZON, target_idx)
    X_val, y_val = make_windows(scale(val_df, scaler), WINDOW_SIZE, HORIZON, target_idx)
    X_test, y_test = make_windows(scale(test_df, scaler), WINDOW_SIZE, HORIZON, target_idx)

    print(f"Train: X={X_train.shape} y={y_train.shape}")
    print(f"Val  : X={X_val.shape} y={y_val.shape}")
    print(f"Test : X={X_test.shape} y={y_test.shape}")

    # 2. Baseline de persistencia: predecir retorno=0 (sin cambio de precio)
    zero_row = np.zeros((1, len(FEATURES)), dtype=np.float32)
    scaled_zero_return = float(scaler.transform(zero_row)[0, target_idx])
    pred_persist = np.full_like(y_test, scaled_zero_return)
    mse_p = float(np.mean((pred_persist - y_test) ** 2))
    mae_p = float(np.mean(np.abs(pred_persist - y_test)))
    print(f"\nBaseline persistencia (retorno=0, escalado={scaled_zero_return:.4f}):")
    print(f"  MSE={mse_p:.6f}  MAE={mae_p:.6f}  RMSE={np.sqrt(mse_p):.6f}")

    metrics = {"persistence": {"mse": mse_p, "mae": mae_p, "rmse": float(np.sqrt(mse_p))}}

    # 3. Entrenar LSTM y Transformer (checkpoints separados: *_logret)
    train_dl, val_dl = make_loaders(X_train, y_train, X_val, y_val)

    print("\n=== Entrenando LSTM (target=Log_Return) ===")
    lstm = LSTMModel()
    train_pytorch(lstm, train_dl, val_dl, "lstm_logret")
    metrics["lstm"] = evaluate_pytorch(lstm, X_test, y_test)
    print(f"  LSTM test: {metrics['lstm']}")
    torch.save(lstm.state_dict(), MODELS_DIR / "lstm_logret_final.pt")

    print("\n=== Entrenando Transformer (target=Log_Return) ===")
    transformer = TransformerModel()
    train_pytorch(transformer, train_dl, val_dl, "transformer_logret")
    metrics["transformer"] = evaluate_pytorch(transformer, X_test, y_test)
    print(f"  Transformer test: {metrics['transformer']}")
    torch.save(transformer.state_dict(), MODELS_DIR / "transformer_logret_final.pt")

    print("\n=== Entrenando Random Forest (target=Log_Return) ===")
    rf = RandomForestModel()
    rf.fit(X_train, y_train)
    rf.save(MODELS_DIR / "rf_logret.pkl")
    pred_rf = rf.predict(X_test)
    mse_rf = float(np.mean((pred_rf - y_test) ** 2))
    metrics["random_forest"] = {
        "mse": mse_rf, "mae": float(np.mean(np.abs(pred_rf - y_test))),
        "rmse": float(np.sqrt(mse_rf)),
    }
    print(f"  RF test: {metrics['random_forest']}")

    (RESULTS_DIR / "logreturn_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"\nMétricas guardadas en results/logreturn_metrics.json "
          f"({time.time()-t0:.1f}s hasta ahora)")

    # 4. Framework de explicabilidad sobre el nuevo target: consistencia + significancia
    reference = X_train.reshape(-1, len(FEATURES)).mean(axis=0)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.sort(rng.choice(len(X_test), size=min(N_EXPLAIN, len(X_test)), replace=False))
    X_explain = X_test[idx]

    models = {"LSTM": lstm, "Transformer": transformer, "RandomForest": rf}
    alphas_dict = {}
    for name, model in models.items():
        mtype = "sklearn" if name == "RandomForest" else "pytorch"
        explainer = ModelAgnosticExplainer(model, mtype, reference, device=DEVICE)
        print(f"\nExplicando [{name}] (target=Log_Return)...")
        alphas = explainer.explain_batch(X_explain, verbose=True)
        alphas_dict[name] = alphas
        np.save(RESULTS_DIR / f"logreturn_alphas_{name}.npy", alphas)

    print("\n=== Consistencia inter-modelo (target=Log_Return) ===")
    consistency = mean_pairwise_consistency(alphas_dict, metric="spearman")
    for (a, b), score in consistency.items():
        print(f"  {a} vs {b}: rho = {score:.4f}")

    print("\n=== Significancia estadística (target=Log_Return) ===")
    sig = per_model_significance(alphas_dict)
    for name, result in sig.items():
        print(f"  {name}: p = {result['p_value']:.2e}")

    results = {
        "target": TARGET_COL,
        "metrics": metrics,
        "consistency_spearman": {f"{a}_{b}": v for (a, b), v in consistency.items()},
        "significance": sig,
    }
    (RESULTS_DIR / "logreturn_explainability_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\nListo. Tiempo total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
