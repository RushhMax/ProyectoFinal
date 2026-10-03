"""
Ablación del orden temporal.

El motor de perturbaciones propuesto (Ecuación 1) sustituye cada celda
(t, v) por una referencia fija r_v, dejando el resto de la ventana en
su valor observado, lo que respeta el orden temporal de la serie. Este
script construye una variante deliberadamente ordenada de forma
incorrecta, "shuffled", que sustituye la celda (t, v) por el valor de
la MISMA variable en OTRO instante t' != t DENTRO DE LA MISMA VENTANA,
elegido al azar. Esto opera exactamente la crítica de la Introducción
a SHAP/LIME, tratar los pasos de tiempo como intercambiables, y con
aproximadamente 50% de probabilidad usa el valor de un instante más
reciente para perturbar uno más antiguo, es decir, inyecta información
futura en un instante pasado.

Se compara FO (determinista, respeta el orden) contra esta variante
"shuffled" (estocástica, viola el orden) en tres frentes, sin
reentrenar ningún modelo:

  1. Determinismo. FO produce siempre la misma matriz para la misma
     ventana. Se mide cuánto varía "shuffled" entre semillas distintas.
  2. Consistencia inter-modelo. Se repite la Tabla de Spearman entre
     LSTM, Transformer y Random Forest bajo la variante "shuffled".
  3. Fidelidad (Delta_alpha, Ecuación 5, alpha=0.3). Se compara FO
     contra "shuffled" sobre las mismas 100 instancias de prueba.

Adicionalmente se repite el benchmark sintético de caja blanca
(Sección de validación con ground truth) con la variante "shuffled",
para verificar si el benchmark sintético es capaz de detectar el daño
de violar el orden temporal.

Uso: python -m experiments.exp_temporal_order_ablation
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import (FEATURES, WINDOW_SIZE, RESULTS_DIR, RANDOM_STATE, N_EXPLAIN)
from data.download import load_raw
from data.preprocessing import prepare_data
from framework.explainer import ModelAgnosticExplainer
from evaluation.consistency import mean_pairwise_consistency
from evaluation.saliency_metrics import faithfulness_shift, compute_all_metrics
from experiments.exp_additional_results import load_close_models
from experiments.exp1_whitebox import generate_ar1, WhiteBoxRegressor, make_predict_fn

T = WINDOW_SIZE
V = len(FEATURES)


# ---------------------------------------------------------------------------
# Variante "shuffled" (viola el orden temporal a propósito)
# ---------------------------------------------------------------------------

def explain_shuffled_batch(predict_fn, X, seed, verbose=False):
    rng = np.random.default_rng(seed)
    N, Tw, Vw = X.shape
    alphas = np.zeros((N, Tw, Vw), dtype=np.float32)
    for i in range(N):
        x = X[i]
        batch = np.tile(x, (Tw * Vw + 1, 1, 1))
        for idx, (t, v) in enumerate(np.ndindex(Tw, Vw)):
            t_prime = t
            while t_prime == t:
                t_prime = int(rng.integers(0, Tw))
            batch[idx + 1, t, v] = x[t_prime, v]
        preds = predict_fn(batch)
        f_orig = float(preds[0])
        alpha = np.abs(preds[1:].astype(np.float32) - f_orig).reshape(Tw, Vw)
        total = alpha.sum()
        if total > 0:
            alpha = alpha / total
        alphas[i] = alpha
        if verbose and (i % 20 == 0 or i == N - 1):
            print(f"    shuffled {i+1}/{N}...", end="\r")
    if verbose:
        print()
    return alphas


def spearman_flat(a, b):
    from scipy.stats import spearmanr
    rho, _ = spearmanr(a.ravel(), b.ravel())
    return float(rho)


# ---------------------------------------------------------------------------
# Parte A. Benchmark sintético de caja blanca (FO vs. shuffled)
# ---------------------------------------------------------------------------

def synthetic_ablation(n_rep=10, N=30, n_svs_unused=None, device="cpu", verbose=True):
    scenarios = {
        "rare_feature": {"A_T_size": T, "A_X_size": 3},
        "rare_time": {"A_T_size": 4, "A_X_size": V},
    }
    method_names = ["FO", "Shuffled"]
    all_results = {}

    for scenario_name, cfg in scenarios.items():
        if verbose:
            print(f"\n  Escenario sintético: {scenario_name}")
        rep_results = {m: [] for m in method_names}

        for rep in range(n_rep):
            rng = np.random.default_rng(rep * 137 + 7)
            salient_features = rng.choice(V, cfg["A_X_size"], replace=False)
            salient_times = rng.choice(T, cfg["A_T_size"], replace=False)
            true_mask = np.zeros((T, V), dtype=bool)
            true_mask[np.ix_(salient_times, salient_features)] = True

            reference = np.zeros(V, dtype=np.float32)
            X_test = np.stack([generate_ar1(T, V, rng) for _ in range(N)])

            wb_model = WhiteBoxRegressor(true_mask).to(device)
            predict_fn = make_predict_fn(wb_model, device)

            from framework.perturbation import TemporalPerturbationEngine
            fo_engine = TemporalPerturbationEngine(predict_fn, reference)
            fo_alphas = fo_engine.explain_batch(X_test, normalize=True, verbose=False)
            rep_results["FO"].append(
                {k: v for k, v in
                 (lambda ms: {k: float(np.mean([m[k] for m in ms])) for k in ms[0]})(
                     [compute_all_metrics(fo_alphas[i], true_mask) for i in range(N)]
                 ).items()}
            )

            shuf_alphas = explain_shuffled_batch(predict_fn, X_test, seed=rep, verbose=False)
            rep_results["Shuffled"].append(
                {k: v for k, v in
                 (lambda ms: {k: float(np.mean([m[k] for m in ms])) for k in ms[0]})(
                     [compute_all_metrics(shuf_alphas[i], true_mask) for i in range(N)]
                 ).items()}
            )

            if verbose:
                print(f"    rep {rep+1}/{n_rep} lista", end="\r")

        scenario_agg = {}
        for method in method_names:
            keys = list(rep_results[method][0].keys())
            agg = {}
            for k in keys:
                vals = [rep_results[method][r][k] for r in range(n_rep)]
                agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            scenario_agg[method] = agg
        all_results[scenario_name] = scenario_agg
        if verbose:
            print()

    return all_results


# ---------------------------------------------------------------------------
# Parte B. S&P 500 real (FO vs. shuffled) sobre los 3 modelos entrenados
# ---------------------------------------------------------------------------

def real_data_ablation(n_seeds=5, verbose=True):
    raw = load_raw()
    X_train, y_train, X_test, y_test, scaler = prepare_data(raw)

    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.sort(rng.choice(len(X_test), size=min(N_EXPLAIN, len(X_test)), replace=False))
    X_explain = X_test[idx]

    reference = X_train.reshape(-1, V).mean(axis=0)
    models = load_close_models()

    alphas_fo = {}
    alphas_shuf_primary = {}
    predict_fns = {}
    faithfulness = {"FO": {}, "Shuffled": {}}

    for name, model in models.items():
        mtype = "sklearn" if name == "RandomForest" else "pytorch"
        explainer = ModelAgnosticExplainer(model, mtype, reference, device="cpu")
        predict_fn = explainer.engine.predict_fn
        predict_fns[name] = predict_fn

        if verbose:
            print(f"  {name}: FO...")
        alphas_fo[name] = explainer.explain_batch(X_explain, verbose=False)

        if verbose:
            print(f"  {name}: shuffled (semilla primaria)...")
        alphas_shuf_primary[name] = explain_shuffled_batch(
            predict_fn, X_explain, seed=RANDOM_STATE, verbose=False)

        faithfulness["FO"][name] = faithfulness_shift(
            predict_fn, X_explain, alphas_fo[name], reference, alpha_frac=0.3)
        faithfulness["Shuffled"][name] = faithfulness_shift(
            predict_fn, X_explain, alphas_shuf_primary[name], reference, alpha_frac=0.3)

    consistency_fo = {f"{a}_{b}": v for (a, b), v in
                       mean_pairwise_consistency(alphas_fo, metric="spearman").items()}
    consistency_shuf = {f"{a}_{b}": v for (a, b), v in
                         mean_pairwise_consistency(alphas_shuf_primary, metric="spearman").items()}

    # Determinismo: K semillas del método shuffled, correlación entre corridas
    reliability = {}
    for name, predict_fn in predict_fns.items():
        if verbose:
            print(f"  {name}: determinismo ({n_seeds} semillas)...")
        runs = []
        for s in range(n_seeds):
            seed = 1000 + s
            runs.append(explain_shuffled_batch(predict_fn, X_explain, seed=seed, verbose=False))
        pair_scores = []
        for a in range(n_seeds):
            for b in range(a + 1, n_seeds):
                for i in range(len(X_explain)):
                    pair_scores.append(spearman_flat(runs[a][i], runs[b][i]))
        reliability[name] = {
            "mean_spearman_between_seeds": float(np.mean(pair_scores)),
            "std_spearman_between_seeds": float(np.std(pair_scores)),
            "n_seeds": n_seeds,
        }

    return {
        "n_instances": int(len(X_explain)),
        "consistency_fo": consistency_fo,
        "consistency_shuffled": consistency_shuf,
        "faithfulness_alpha_0.3": faithfulness,
        "reliability_shuffled_across_seeds": reliability,
        "fo_determinism": "1.000 por construcción (misma ventana, misma referencia fija, sin aleatoriedad)",
    }


def main():
    t0 = time.time()
    print("=== Parte A: benchmark sintético (FO vs. shuffled) ===")
    synth = synthetic_ablation(verbose=True)

    print("\n=== Parte B: S&P 500 real (FO vs. shuffled) ===")
    real = real_data_ablation(verbose=True)

    out = {"synthetic": synth, "real_sp500": real}
    out_path = RESULTS_DIR / "temporal_order_ablation.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\nGuardado en {out_path}")
    print(f"Tiempo total: {elapsed:.1f}s")

    print("\n--- Resumen sintético (AUP) ---")
    for scenario, methods in synth.items():
        print(f"  {scenario}: FO AUP={methods['FO']['AUP']['mean']:.4f} "
              f"vs Shuffled AUP={methods['Shuffled']['AUP']['mean']:.4f}")

    print("\n--- Resumen consistencia inter-modelo (S&P 500 real) ---")
    print(f"  FO:       {real['consistency_fo']}")
    print(f"  Shuffled: {real['consistency_shuffled']}")

    print("\n--- Resumen fidelidad (Delta_0.3) ---")
    print(f"  FO:       {real['faithfulness_alpha_0.3']['FO']}")
    print(f"  Shuffled: {real['faithfulness_alpha_0.3']['Shuffled']}")

    print("\n--- Resumen determinismo (shuffled, correlación entre semillas) ---")
    for name, r in real["reliability_shuffled_across_seeds"].items():
        print(f"  {name}: {r['mean_spearman_between_seeds']:.3f} "
              f"+/- {r['std_spearman_between_seeds']:.3f}")


if __name__ == "__main__":
    main()
