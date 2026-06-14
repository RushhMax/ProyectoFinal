"""
Generación y evaluación de explicaciones con el framework propuesto.

Uso: python explain.py

Requiere haber ejecutado train.py previamente.
"""
import numpy as np
import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (N_EXPLAIN, FEATURES, RESULTS_DIR, MODELS_DIR,
                    RANDOM_STATE, WINDOW_SIZE)
from data.download import load_raw
from data.preprocessing import prepare_data, load_scaler
from models.lstm import LSTMModel
from models.transformer import TransformerModel
from models.random_forest import RandomForestModel
from framework.explainer import ModelAgnosticExplainer
from evaluation.consistency import mean_pairwise_consistency
from evaluation.significance import per_model_significance
from visualization.plots import (plot_heatmaps_comparison,
                                  plot_variable_importance,
                                  plot_temporal_importance)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(RANDOM_STATE)


def load_models(X_train_shape):
    """Carga los tres modelos desde disco."""
    lstm = LSTMModel()
    lstm.load_state_dict(torch.load(MODELS_DIR / "lstm_best.pt",
                                    map_location=DEVICE, weights_only=True))
    lstm.eval()

    transformer = TransformerModel()
    transformer.load_state_dict(torch.load(MODELS_DIR / "transformer_best.pt",
                                            map_location=DEVICE, weights_only=True))
    transformer.eval()

    rf = RandomForestModel.load()

    return {"LSTM": lstm, "Transformer": transformer, "RandomForest": rf}


def main():
    # 1. Datos
    df = load_raw()
    X_train, y_train, X_test, y_test, scaler = prepare_data(df)

    # Valor de referencia = media de entrenamiento por variable (en escala normalizada)
    reference = X_train.reshape(-1, len(FEATURES)).mean(axis=0)

    # Subconjunto de test para explicar
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_test), size=min(N_EXPLAIN, len(X_test)), replace=False)
    idx = np.sort(idx)
    X_explain = X_test[idx]

    # 2. Cargar modelos
    print("Cargando modelos...")
    models = load_models(X_train.shape)

    # 3. Crear explainers y generar matrices de importancia
    alphas_dict = {}
    for name, model in models.items():
        mtype = "sklearn" if name == "RandomForest" else "pytorch"
        explainer = ModelAgnosticExplainer(model, mtype, reference, device=DEVICE)
        print(f"\nExplicando [{name}]...")
        alphas = explainer.explain_batch(X_explain, verbose=True)
        alphas_dict[name] = alphas
        np.save(RESULTS_DIR / f"alphas_{name}.npy", alphas)
        print(f"  Guardado: results/alphas_{name}.npy  shape={alphas.shape}")

    # 4. Evaluación de consistencia
    print("\n=== Consistencia entre modelos ===")
    consistency = mean_pairwise_consistency(alphas_dict, metric="spearman")
    for (a, b), score in consistency.items():
        print(f"  {a} vs {b}: ρ = {score:.4f}")

    # 5. Significancia estadística
    print("\n=== Significancia estadística (Mann-Whitney vs. azar) ===")
    sig = per_model_significance(alphas_dict)
    for name, result in sig.items():
        p = result["p_value"]
        flag = "✓ significativo" if p < 0.05 else "✗ no significativo"
        print(f"  {name}: p = {p:.2e}  {flag}")

    # 6. Guardar resultados numéricos
    results = {
        "consistency_spearman": {f"{a}_{b}": v for (a, b), v in consistency.items()},
        "significance": sig,
    }
    (RESULTS_DIR / "explainability_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # 7. Visualizaciones
    print("\nGenerando visualizaciones...")
    plot_heatmaps_comparison(
        alphas_dict,
        save_path=RESULTS_DIR / "heatmaps_comparison.png"
    )
    plot_variable_importance(
        alphas_dict,
        save_path=RESULTS_DIR / "variable_importance.png"
    )
    plot_temporal_importance(
        alphas_dict,
        save_path=RESULTS_DIR / "temporal_importance.png"
    )
    print("Listo. Resultados en results/")


if __name__ == "__main__":
    main()
