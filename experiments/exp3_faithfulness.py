"""
Experimento 3 — Faithfulness en datos reales S&P 500
(análogo a Exp. 3.3 de Dynamask, Crabbé & van der Schaar, 2021).

Evaluación sin ground truth
---------------------------
Al no conocer la importancia verdadera en datos reales, se evalúa la
"faithfulness" (fidelidad) de cada método reemplazando las celdas más
importantes por su valor de referencia y midiendo cuánto cambia la
predicción.  Una buena explicación → mayor cambio al perturbar.

Métricas
--------
- MAE shift  : E[|f(X) - f(X̃)|]  (↑ better, análogo al CE de Dynamask)
- Dir. cons. : fracción donde sign(f(X)) = sign(f(X̃))  (↓ better, análogo ACC)

Métodos comparados
------------------
- FO  : Feature Occlusion (= método propuesto TPE)
- FP  : Feature Permutation
- IG  : Integrated Gradients (solo para LSTM y Transformer)
- SVS : Shapley Value Sampling

Para cada fracción α ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6} se calcula MAE shift
y Dir. cons., generando curvas comparables con las Figuras 4 y 5 de Dynamask.

Uso
---
    python -m experiments.exp3_faithfulness
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import (FEATURES, RESULTS_DIR, MODELS_DIR, WINDOW_SIZE,
                    RANDOM_STATE, N_EXPLAIN)
from data.download import load_raw
from data.preprocessing import prepare_data
from models.lstm import LSTMModel
from models.transformer import TransformerModel
from models.random_forest import RandomForestModel
from framework.perturbation import TemporalPerturbationEngine
from baselines.baseline_explainers import FPExplainer, IGExplainer, SVSExplainer
from evaluation.saliency_metrics import faithfulness_curve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA_FRACS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------

def load_models():
    for ckpt in ["lstm_best.pt", "transformer_best.pt"]:
        path = MODELS_DIR / ckpt
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint no encontrado: {path}\n"
                "Ejecuta primero: python train.py"
            )

    lstm = LSTMModel()
    lstm.load_state_dict(torch.load(MODELS_DIR / "lstm_best.pt",
                                    map_location=DEVICE, weights_only=True))
    lstm.eval()

    transformer = TransformerModel()
    transformer.load_state_dict(torch.load(MODELS_DIR / "transformer_best.pt",
                                            map_location=DEVICE,
                                            weights_only=True))
    transformer.eval()

    rf = RandomForestModel.load()
    return {"LSTM": lstm, "Transformer": transformer, "RandomForest": rf}


def make_predict_fn(name: str, model, device: str):
    if name == "RandomForest":
        return model.predict
    model_ref = model
    model_ref.eval()

    def fn(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).to(device)
            return model_ref(t).cpu().numpy()

    return fn


# ---------------------------------------------------------------------------
# Generación de explicaciones para todos los métodos
# ---------------------------------------------------------------------------

def generate_all_explanations(name: str,
                               model,
                               predict_fn,
                               X_test: np.ndarray,
                               X_train: np.ndarray,
                               reference: np.ndarray,
                               verbose: bool = True) -> dict:
    """
    Genera matrices alpha (N, T, V) para FO, FP, IG (si aplica) y SVS.
    Retorna dict método → (N, T, V).
    """
    T, V = X_test.shape[1], X_test.shape[2]
    alphas_by_method: dict[str, np.ndarray] = {}

    if verbose:
        print(f"\n  [FO — método propuesto]")
    fo = TemporalPerturbationEngine(predict_fn, reference)
    alphas_by_method["FO"] = fo.explain_batch(X_test, verbose=verbose)

    if verbose:
        print(f"\n  [FP — Feature Permutation]")
    fp = FPExplainer(predict_fn, reference,
                     X_background=X_train, seed=RANDOM_STATE)
    alphas_by_method["FP"] = fp.explain_batch(X_test, verbose=verbose)

    # IG sólo para modelos diferenciables (PyTorch)
    if name != "RandomForest":
        if verbose:
            print(f"\n  [IG — Integrated Gradients]")
        ig = IGExplainer(model, reference, device=DEVICE, n_steps=50)
        alphas_by_method["IG"] = ig.explain_batch(X_test, verbose=verbose)
    else:
        if verbose:
            print(f"\n  [IG — omitido para RandomForest]")
        alphas_by_method["IG"] = None

    # SVS en subconjunto más pequeño por coste computacional
    n_svs = min(10, len(X_test))
    if verbose:
        print(f"\n  [SVS — Shapley Value Sampling sobre {n_svs} muestras]")
    svs = SVSExplainer(predict_fn, reference, n_samples=15, seed=RANDOM_STATE)
    svs_alphas = svs.explain_batch(X_test[:n_svs], verbose=verbose)
    # Rellena el resto con NaN para indicar muestras no evaluadas
    if n_svs < len(X_test):
        pad = np.full((len(X_test) - n_svs, T, V), np.nan, dtype=np.float32)
        alphas_by_method["SVS"] = np.concatenate([svs_alphas, pad], axis=0)
    else:
        alphas_by_method["SVS"] = svs_alphas

    return alphas_by_method


# ---------------------------------------------------------------------------
# Evaluación faithfulness
# ---------------------------------------------------------------------------

def compute_faithfulness_all(alphas_by_method: dict,
                              predict_fn,
                              X_test: np.ndarray,
                              reference: np.ndarray,
                              fracs: np.ndarray = ALPHA_FRACS) -> dict:
    """
    Calcula curvas de faithfulness para cada método.
    Para SVS, usa sólo las muestras sin NaN.
    """
    curves = {}
    for method, alphas in alphas_by_method.items():
        if alphas is None:
            curves[method] = None
            continue

        # Para SVS: solo muestras con valores válidos
        if method == "SVS":
            valid = ~np.isnan(alphas).any(axis=(1, 2))
            if valid.sum() == 0:
                curves[method] = None
                continue
            alphas_valid = alphas[valid]
            X_valid = X_test[valid]
        else:
            alphas_valid = alphas
            X_valid = X_test

        curves[method] = faithfulness_curve(
            predict_fn, X_valid, alphas_valid, reference, fracs
        )

    return curves


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def plot_faithfulness_curves(all_model_curves: dict,
                              fracs: np.ndarray,
                              save_dir: Path) -> None:
    """
    Genera los plots análogos a las Figuras 4 y 5 de Dynamask.
    Un subplot por modelo (LSTM, Transformer, RF).
    """
    model_names = list(all_model_curves.keys())
    n_models = len(model_names)

    colors = {"FO": "#1f77b4", "FP": "#ff7f0e",
              "IG": "#2ca02c", "SVS": "#d62728"}
    markers = {"FO": "o", "FP": "s", "IG": "^", "SVS": "D"}
    linestyles = {"FO": "-", "FP": "--", "IG": "-.", "SVS": ":"}

    # --- MAE Shift (Figura 4 analog) ---
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, mname in zip(axes, model_names):
        curves = all_model_curves[mname]
        for method, curve in curves.items():
            if curve is None:
                continue
            ax.plot(curve["fracs"], curve["mae_shift"],
                    label=method, color=colors.get(method, "black"),
                    marker=markers.get(method, "o"),
                    linestyle=linestyles.get(method, "-"), linewidth=2)
        ax.set_title(mname, fontsize=12)
        ax.set_xlabel("Fracción de celdas perturbadas (α)", fontsize=10)
        ax.set_ylabel("MAE Shift ↑", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Faithfulness — MAE Shift al perturbar top-α celdas\n"
                 "(mayor es mejor: la perturbación afecta más la predicción)",
                 fontsize=11)
    plt.tight_layout()
    path_shift = save_dir / "exp3_mae_shift.png"
    fig.savefig(path_shift, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path_shift}")

    # --- Direction Consistency (Figura 5 analog) ---
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, mname in zip(axes, model_names):
        curves = all_model_curves[mname]
        for method, curve in curves.items():
            if curve is None:
                continue
            ax.plot(curve["fracs"], curve["dir_consistency"],
                    label=method, color=colors.get(method, "black"),
                    marker=markers.get(method, "o"),
                    linestyle=linestyles.get(method, "-"), linewidth=2)
        ax.set_title(mname, fontsize=12)
        ax.set_xlabel("Fracción de celdas perturbadas (α)", fontsize=10)
        ax.set_ylabel("Consistencia de dirección ↓", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Faithfulness — Consistencia de dirección tras perturbar top-α celdas\n"
                 "(menor es mejor: más predicciones cambian de dirección)",
                 fontsize=11)
    plt.tight_layout()
    path_dir = save_dir / "exp3_dir_consistency.png"
    fig.savefig(path_dir, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path_dir}")


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------

def run_faithfulness_experiment(verbose: bool = True) -> dict:
    """
    Ejecuta el Experimento 3 sobre los tres modelos entrenados.
    Retorna dict con curvas de faithfulness por modelo y método.
    """
    # ---- Datos ----
    df = load_raw()
    X_train, y_train, X_test, y_test, scaler = prepare_data(df)
    V = len(FEATURES)
    reference = X_train.reshape(-1, V).mean(axis=0).astype(np.float32)

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_test), size=min(N_EXPLAIN, len(X_test)), replace=False)
    idx = np.sort(idx)
    X_explain = X_test[idx].astype(np.float32)

    if verbose:
        print(f"Datos: X_explain shape = {X_explain.shape}")
        print(f"Referencia: media de entrenamiento por variable (shape {reference.shape})")

    # ---- Modelos ----
    if verbose:
        print("\nCargando modelos...")
    models = load_models()

    all_model_curves = {}

    for model_name, model in models.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Modelo: {model_name}")
            print(f"{'='*60}")

        predict_fn = make_predict_fn(model_name, model, DEVICE)

        # Genera alphas para todos los métodos
        alphas_by_method = generate_all_explanations(
            model_name, model, predict_fn,
            X_explain, X_train, reference, verbose=verbose
        )

        # Calcula curvas faithfulness
        if verbose:
            print(f"\n  Calculando curvas faithfulness...")
        curves = compute_faithfulness_all(
            alphas_by_method, predict_fn, X_explain, reference
        )
        all_model_curves[model_name] = curves

        # Reporte numérico
        if verbose:
            print(f"\n  MAE Shift por método y fracción:")
            print(f"  {'Método':<10}", end="")
            for f in ALPHA_FRACS:
                print(f"  α={f:.1f}", end="")
            print()
            for method, curve in curves.items():
                if curve is None:
                    continue
                print(f"  {method:<10}", end="")
                for v in curve["mae_shift"]:
                    print(f"  {v:.4f}", end="")
                print()

    # ---- Visualización ----
    if verbose:
        print("\nGenerando gráficas...")
    plot_faithfulness_curves(all_model_curves, ALPHA_FRACS, RESULTS_DIR)

    # ---- Guardar JSON ----
    # Convertir numpy arrays a listas para JSON
    serializable = {}
    for model_name, curves in all_model_curves.items():
        serializable[model_name] = {}
        for method, curve in curves.items():
            if curve is None:
                serializable[model_name][method] = None
            else:
                serializable[model_name][method] = {
                    "fracs": [float(f) for f in curve["fracs"]],
                    "mae_shift": [float(v) for v in curve["mae_shift"]],
                    "dir_consistency": [float(v) for v in curve["dir_consistency"]],
                }

    out_path = RESULTS_DIR / "exp3_faithfulness_results.json"
    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    if verbose:
        print(f"Resultados guardados en: {out_path}")

    return all_model_curves


if __name__ == "__main__":
    print("Experimento 3 — Faithfulness en S&P 500")
    print("(análogo a Exp. 3.3 de Dynamask)\n")

    t0 = time.time()
    curves = run_faithfulness_experiment(verbose=True)
    elapsed = time.time() - t0
    print(f"\nTiempo total: {elapsed:.1f}s")
