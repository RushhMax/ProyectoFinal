"""
Análisis adicionales que reutilizan los checkpoints ya entrenados
(no reentrena ningún modelo):

  1. Exactitud direccional (target=Close y target=Log_Return).
  2. Sensibilidad del framework al vector de referencia (media vs.
     mediana vs. cero).
  3. Estabilidad de las matrices de importancia por régimen de mercado
     dentro del conjunto de prueba real (bajista 2022 vs. alcista
     2023-2024; el período de prueba es 2022-01-04 a 2024-12-30, no
     incluye el crash de COVID de 2020).

Uso: python -m experiments.exp_additional_results
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from config import (FEATURES, WINDOW_SIZE, HORIZON, TRAIN_RATIO,
                     MODELS_DIR, RESULTS_DIR, RANDOM_STATE, N_EXPLAIN)
from data.download import load_raw
from data.features import add_indicators
from data.preprocessing import (split_temporal, load_scaler, scale,
                                 make_windows, inverse_target, prepare_data)
from models.lstm import LSTMModel
from models.transformer import TransformerModel
from models.random_forest import RandomForestModel
from framework.explainer import ModelAgnosticExplainer
from evaluation.consistency import mean_pairwise_consistency, spearman_correlation

DEVICE = "cpu"
np.random.seed(RANDOM_STATE)


def load_close_models():
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
    return {"LSTM": lstm, "Transformer": transformer, "RandomForest": rf}


def predict(model, name, X):
    if name == "RandomForest":
        return model.predict(X)
    with torch.no_grad():
        return model(torch.tensor(X)).cpu().numpy()


# ---------------------------------------------------------------------------
# 1. Exactitud direccional
# ---------------------------------------------------------------------------

def directional_accuracy(df, dev_df, test_df, scaler):
    results = {"target_close": {}, "target_logreturn": {}}

    close_idx = FEATURES.index("Close")
    X_test_c, y_test_c = make_windows(scale(test_df, scaler), WINDOW_SIZE, HORIZON, close_idx)
    last_close = inverse_target(X_test_c[:, -1, close_idx], scaler, close_idx)
    y_true_close = inverse_target(y_test_c, scaler, close_idx)
    true_dir_c = np.sign(y_true_close - last_close)

    models_c = load_close_models()
    for name, model in models_c.items():
        pred = predict(model, name, X_test_c)
        pred_close = inverse_target(pred, scaler, close_idx)
        pred_dir = np.sign(pred_close - last_close)
        mask = true_dir_c != 0
        results["target_close"][name] = float(np.mean(pred_dir[mask] == true_dir_c[mask]))

    ret_idx = FEATURES.index("Log_Return")
    X_test_r, y_test_r = make_windows(scale(test_df, scaler), WINDOW_SIZE, HORIZON, ret_idx)
    y_true_ret = inverse_target(y_test_r, scaler, ret_idx)
    true_dir_r = np.sign(y_true_ret)

    models_r = load_logret_models()
    for name, model in models_r.items():
        pred = predict(model, name, X_test_r)
        pred_ret = inverse_target(pred, scaler, ret_idx)
        pred_dir = np.sign(pred_ret)
        mask = true_dir_r != 0
        results["target_logreturn"][name] = float(np.mean(pred_dir[mask] == true_dir_r[mask]))

    results["n_test_windows"] = int(len(y_test_c))
    return results


# ---------------------------------------------------------------------------
# 2. Sensibilidad al vector de referencia
# ---------------------------------------------------------------------------

def reference_sensitivity(raw):
    X_train, y_train, X_test, y_test, scaler = prepare_data(raw)
    V = len(FEATURES)

    references = {
        "mean": X_train.reshape(-1, V).mean(axis=0),
        "median": np.median(X_train.reshape(-1, V), axis=0),
        "zero": np.zeros(V, dtype=np.float32),
    }

    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.sort(rng.choice(len(X_test), size=min(N_EXPLAIN, len(X_test)), replace=False))
    X_explain = X_test[idx]

    models = load_close_models()
    alphas_by_ref = {}
    for ref_name, ref_vec in references.items():
        alphas_dict = {}
        for name, model in models.items():
            mtype = "sklearn" if name == "RandomForest" else "pytorch"
            explainer = ModelAgnosticExplainer(model, mtype, ref_vec, device=DEVICE)
            alphas_dict[name] = explainer.explain_batch(X_explain, verbose=False)
        alphas_by_ref[ref_name] = alphas_dict
        print(f"  Referencia '{ref_name}' lista.")

    stability = {}
    for name in models:
        base = alphas_by_ref["mean"][name]
        stability[name] = {}
        for ref_name in ["median", "zero"]:
            other = alphas_by_ref[ref_name][name]
            corrs = [spearman_correlation(base[i], other[i]) for i in range(len(base))]
            stability[name][f"mean_vs_{ref_name}"] = float(np.mean(corrs))

    consistency_by_ref = {}
    for ref_name, alphas_dict in alphas_by_ref.items():
        consistency_by_ref[ref_name] = {
            f"{a}_{b}": v for (a, b), v in
            mean_pairwise_consistency(alphas_dict, metric="spearman").items()
        }

    return {"stability_vs_mean_reference": stability,
            "consistency_by_reference": consistency_by_ref}


# ---------------------------------------------------------------------------
# 3. Régimen de mercado
# ---------------------------------------------------------------------------

def market_regime(raw, dev_df, test_df, scaler):
    X_train, y_train, X_test, y_test, _ = prepare_data(raw)

    n_windows = len(test_df) - WINDOW_SIZE - HORIZON + 1
    target_dates = pd.DatetimeIndex(
        [test_df.index[i + WINDOW_SIZE + HORIZON - 1] for i in range(n_windows)]
    )

    regimes = {
        "bajista_2022": (target_dates >= "2022-01-01") & (target_dates <= "2022-10-31"),
        "alcista_2023_2024": (target_dates >= "2023-01-01") & (target_dates <= "2024-12-31"),
    }
    for name, mask in regimes.items():
        print(f"  Régimen '{name}': {mask.sum()} ventanas")

    reference = X_train.reshape(-1, len(FEATURES)).mean(axis=0)
    models = load_close_models()

    results = {}
    for regime_name, mask in regimes.items():
        X_regime = X_test[mask]
        alphas_dict = {}
        for name, model in models.items():
            mtype = "sklearn" if name == "RandomForest" else "pytorch"
            explainer = ModelAgnosticExplainer(model, mtype, reference, device=DEVICE)
            alphas_dict[name] = explainer.explain_batch(X_regime, verbose=False)
        consistency = mean_pairwise_consistency(alphas_dict, metric="spearman")
        var_importance = {
            name: {FEATURES[v]: float(a) for v, a in
                   enumerate(alphas.mean(axis=(0, 1)))}
            for name, alphas in alphas_dict.items()
        }
        results[regime_name] = {
            "n_windows": int(mask.sum()),
            "consistency_spearman": {f"{a}_{b}": v for (a, b), v in consistency.items()},
            "mean_importance_by_variable": var_importance,
        }
        print(f"  Régimen '{regime_name}' listo.")

    return results


def main():
    raw = load_raw()
    df = add_indicators(raw)
    dev_df, test_df = split_temporal(df, TRAIN_RATIO)
    scaler = load_scaler()

    print("=== 1. Exactitud direccional ===")
    dir_acc = directional_accuracy(df, dev_df, test_df, scaler)
    print(json.dumps(dir_acc, indent=2))

    print("\n=== 2. Sensibilidad al vector de referencia ===")
    ref_sens = reference_sensitivity(raw)
    print(json.dumps(ref_sens, indent=2))

    print("\n=== 3. Régimen de mercado ===")
    regime = market_regime(raw, dev_df, test_df, scaler)
    print(json.dumps({k: {"n_windows": v["n_windows"],
                           "consistency_spearman": v["consistency_spearman"]}
                       for k, v in regime.items()}, indent=2))

    out = {
        "directional_accuracy": dir_acc,
        "reference_sensitivity": ref_sens,
        "market_regime": regime,
    }
    (RESULTS_DIR / "additional_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("\nGuardado en results/additional_results.json")


if __name__ == "__main__":
    main()
