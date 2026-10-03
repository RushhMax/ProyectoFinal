"""
Vector de referencia condicionado a la hoja para Random Forest.

Compara la referencia global fija (media de entrenamiento, usada hoy
para los 3 modelos) contra una referencia condicionada a la hoja,
calculada por instancia a partir de los ejemplos de entrenamiento que
caen en la misma hoja que la instancia en cada árbol del ensamble
(framework/leaf_reference.py). No reentrena ningún modelo.

Dos partes:
  A. Benchmark sintético con ground truth conocido (se entrena un RF
     real sobre datos AR(1), a diferencia de exp1_whitebox.py que usa
     directamente la función de caja blanca como "modelo").
  B. S&P 500 real, reutilizando saved_models/rf_logret.pkl y los
     checkpoints de LSTM/Transformer (target Log_Return).

Uso: python -m experiments.exp_rf_leaf_reference
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

from config import FEATURES, WINDOW_SIZE, MODELS_DIR, RESULTS_DIR, RANDOM_STATE, N_EXPLAIN
from framework.perturbation import TemporalPerturbationEngine
from framework.leaf_reference import leaf_conditional_reference
from evaluation.saliency_metrics import compute_all_metrics
from evaluation.consistency import spearman_correlation
from experiments.exp1_whitebox import generate_ar1
from models.lstm import LSTMModel
from models.transformer import TransformerModel
from models.random_forest import RandomForestModel
from data.download import load_raw
from data.preprocessing import prepare_data

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Parte A: benchmark sintético con un RF real entrenado
# ---------------------------------------------------------------------------

def run_synthetic(T: int = 20, V: int = 15,
                   n_train: int = 3000, n_test: int = 30,
                   n_salient_features: int = 3,
                   n_rep: int = 5, verbose: bool = True) -> dict:
    """
    Entrena un RandomForestRegressor real sobre f(X) = Sum_{(t,v) in A} x_{t,v}^2
    (escenario "rare_feature": todos los tiempos salientes, pocas features),
    y compara AUP/AUR de FO usando referencia global (media) vs. referencia
    condicionada a la hoja.
    """
    results = {"FO_media_global": [], "FO_hoja": []}

    for rep in range(n_rep):
        rng = np.random.default_rng(rep * 137 + 7)
        salient_features = rng.choice(V, n_salient_features, replace=False)
        true_mask = np.zeros((T, V), dtype=bool)
        true_mask[:, salient_features] = True

        def f(X):
            return (X[:, :, salient_features] ** 2).sum(axis=(1, 2))

        X_train = np.stack([generate_ar1(T, V, rng) for _ in range(n_train)])
        y_train = f(X_train)
        X_test = np.stack([generate_ar1(T, V, rng) for _ in range(n_test)])

        rf = RandomForestRegressor(n_estimators=200, random_state=rep, n_jobs=-1)
        X_train_flat = X_train.reshape(n_train, -1)
        rf.fit(X_train_flat, y_train)

        def predict_fn(X):
            return rf.predict(X.reshape(X.shape[0], -1)).astype(np.float32)

        # ---- Referencia global (media de entrenamiento) ----
        ref_global = X_train.reshape(-1, V).mean(axis=0)
        engine = TemporalPerturbationEngine(predict_fn, ref_global)
        alphas_global = engine.explain_batch(X_test, normalize=True, verbose=False)
        m_global = compute_all_metrics(
            np.mean(alphas_global, axis=0), true_mask
        )
        # métrica por instancia promediada (más robusto que sobre el promedio)
        m_global = {k: float(np.mean([compute_all_metrics(alphas_global[i], true_mask)[k]
                                       for i in range(n_test)]))
                    for k in m_global}
        results["FO_media_global"].append(m_global)

        # ---- Referencia condicionada a la hoja ----
        alphas_leaf = []
        for i in range(n_test):
            ref_leaf = leaf_conditional_reference(rf, X_test[i].reshape(-1),
                                                   X_train_flat, T, V)
            eng_i = TemporalPerturbationEngine(predict_fn, ref_leaf)
            alphas_leaf.append(eng_i.explain(X_test[i], normalize=True))
        alphas_leaf = np.stack(alphas_leaf)
        m_leaf = {k: float(np.mean([compute_all_metrics(alphas_leaf[i], true_mask)[k]
                                     for i in range(n_test)]))
                  for k in m_global}
        results["FO_hoja"].append(m_leaf)

        if verbose:
            print(f"  Rep {rep+1}/{n_rep}: AUP media_global={m_global['AUP']:.4f} "
                  f"vs AUP hoja={m_leaf['AUP']:.4f} | "
                  f"AUR media_global={m_global['AUR']:.4f} "
                  f"vs AUR hoja={m_leaf['AUR']:.4f}")

    agg = {}
    for method in results:
        keys = results[method][0].keys()
        agg[method] = {k: {"mean": float(np.mean([r[k] for r in results[method]])),
                            "std": float(np.std([r[k] for r in results[method]]))}
                       for k in keys}
    return agg


# ---------------------------------------------------------------------------
# Parte B: S&P 500 real (target Log_Return, checkpoints ya entrenados)
# ---------------------------------------------------------------------------

def load_logret_models():
    lstm = LSTMModel()
    lstm.load_state_dict(torch.load(MODELS_DIR / "lstm_logret_best.pt",
                                     map_location=DEVICE, weights_only=True))
    lstm.eval()
    transformer = TransformerModel()
    transformer.load_state_dict(torch.load(MODELS_DIR / "transformer_logret_best.pt",
                                            map_location=DEVICE, weights_only=True))
    transformer.eval()
    rf = RandomForestModel.load(MODELS_DIR / "rf_logret.pkl")
    return lstm, transformer, rf


def temporal_ratio(alphas: np.ndarray) -> float:
    """Razón máx/mín de la importancia media por paso de tiempo (T,)."""
    by_t = alphas.mean(axis=(0, 2))
    mn = by_t.min()
    return float(by_t.max() / mn) if mn > 0 else float("inf")


def run_real_data(verbose: bool = True) -> dict:
    raw = load_raw()
    X_train, y_train, X_test, y_test, scaler = prepare_data(raw)
    V = len(FEATURES)
    T = WINDOW_SIZE

    lstm, transformer, rf_wrapped = load_logret_models()
    rf = rf_wrapped.model  # RandomForestRegressor de sklearn

    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.sort(rng.choice(len(X_test), size=min(N_EXPLAIN, len(X_test)), replace=False))
    X_explain = X_test[idx]
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    def lstm_fn(X):
        with torch.no_grad():
            return lstm(torch.tensor(X, dtype=torch.float32)).cpu().numpy()

    def transformer_fn(X):
        with torch.no_grad():
            return transformer(torch.tensor(X, dtype=torch.float32)).cpu().numpy()

    ref_global = X_train.reshape(-1, V).mean(axis=0)

    if verbose:
        print("  Explicando LSTM y Transformer (referencia global, sin cambios)...")
    lstm_engine = TemporalPerturbationEngine(lstm_fn, ref_global)
    alphas_lstm = lstm_engine.explain_batch(X_explain, normalize=True, verbose=False)
    tf_engine = TemporalPerturbationEngine(transformer_fn, ref_global)
    alphas_tf = tf_engine.explain_batch(X_explain, normalize=True, verbose=False)

    if verbose:
        print("  Explicando Random Forest con referencia global (media)...")
    rf_engine = TemporalPerturbationEngine(rf_wrapped.predict, ref_global)
    alphas_rf_global = rf_engine.explain_batch(X_explain, normalize=True, verbose=False)

    if verbose:
        print("  Explicando Random Forest con referencia condicionada a la hoja...")
    alphas_rf_leaf = []
    for i in range(len(X_explain)):
        if verbose and i % 10 == 0:
            print(f"    {i+1}/{len(X_explain)}...", end="\r")
        ref_leaf = leaf_conditional_reference(rf, X_explain[i].reshape(-1),
                                               X_train_flat, T, V)
        eng_i = TemporalPerturbationEngine(rf_wrapped.predict, ref_leaf)
        alphas_rf_leaf.append(eng_i.explain(X_explain[i], normalize=True))
    if verbose:
        print()
    alphas_rf_leaf = np.stack(alphas_rf_leaf)

    consistency = {
        "global_reference": {
            "LSTM_RF": float(np.mean([spearman_correlation(alphas_lstm[i], alphas_rf_global[i])
                                       for i in range(len(X_explain))])),
            "Transformer_RF": float(np.mean([spearman_correlation(alphas_tf[i], alphas_rf_global[i])
                                              for i in range(len(X_explain))])),
        },
        "leaf_reference": {
            "LSTM_RF": float(np.mean([spearman_correlation(alphas_lstm[i], alphas_rf_leaf[i])
                                       for i in range(len(X_explain))])),
            "Transformer_RF": float(np.mean([spearman_correlation(alphas_tf[i], alphas_rf_leaf[i])
                                              for i in range(len(X_explain))])),
        },
    }

    temporal_concentration = {
        "RF_global_reference": temporal_ratio(alphas_rf_global),
        "RF_leaf_reference": temporal_ratio(alphas_rf_leaf),
        "LSTM_reference": temporal_ratio(alphas_lstm),
        "Transformer_reference": temporal_ratio(alphas_tf),
    }

    # Similitud entre las dos versiones de RF (para saber si el cambio es
    # drástico o una variación moderada sobre el mismo patrón)
    rf_self_consistency = float(np.mean(
        [spearman_correlation(alphas_rf_global[i], alphas_rf_leaf[i])
         for i in range(len(X_explain))]
    ))

    return {
        "n_explained": int(len(X_explain)),
        "consistency_spearman": consistency,
        "temporal_concentration_ratio": temporal_concentration,
        "rf_global_vs_leaf_self_consistency": rf_self_consistency,
    }


def main():
    print("=== Parte A: benchmark sintético (RF real entrenado) ===")
    t0 = time.time()
    synth = run_synthetic()
    print(json.dumps(synth, indent=2))
    print(f"Tiempo: {time.time()-t0:.1f}s")

    print("\n=== Parte B: S&P 500 real (rf_logret.pkl, sin reentrenar) ===")
    t0 = time.time()
    real = run_real_data()
    print(json.dumps(real, indent=2))
    print(f"Tiempo: {time.time()-t0:.1f}s")

    out = {"synthetic": synth, "real_sp500": real}
    out_path = RESULTS_DIR / "rf_leaf_reference_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
