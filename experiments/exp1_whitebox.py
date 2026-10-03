"""
Experimento 1 — White-box regressor con importancia conocida
(análogo a Exp. 3.1 de Dynamask, Crabbé & van der Schaar, 2021).

Configuración
-------------
Se usa un regresor de caja blanca cuya predicción depende únicamente de
un subconjunto A = A_T × A_X ⊂ [0:T] × [0:V] de celdas salientes:

    f(X) = Σ_{t ∈ A_T, v ∈ A_X} x_{t,v}²

Las entradas se generan con un proceso AR(1) por variable.
Se evalúan los siguientes métodos:
    - FO  : Feature Occlusion (método propuesto = TPE)
    - FP  : Feature Permutation
    - IG  : Integrated Gradients (modelo PyTorch diferenciable)
    - SVS : Shapley Value Sampling

Escenarios
----------
1. "rare_feature" : pocas features salientes, todos los tiempos salientes
2. "rare_time"    : pocas ventanas temporales salientes, todas las features

Salida
------
Dict con tabla de resultados media ± std para cada método y escenario,
compatible con el formato de las Tablas 1 y 2 de Dynamask.

Uso
---
    python -m experiments.exp1_whitebox
    # o importar run_whitebox_experiments() desde otro script
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import json

from framework.perturbation import TemporalPerturbationEngine
from baselines.baseline_explainers import FPExplainer, IGExplainer, SVSExplainer
from evaluation.saliency_metrics import compute_all_metrics

# ---------------------------------------------------------------------------
# Generación de datos AR(1)
# ---------------------------------------------------------------------------

def generate_ar1(T: int, V: int, rng: np.random.Generator) -> np.ndarray:
    """Genera una ventana (T, V) con V procesos AR(1) independientes."""
    coef = rng.uniform(0.3, 0.7, V).astype(np.float32)
    X = np.zeros((T, V), dtype=np.float32)
    X[0] = rng.standard_normal(V).astype(np.float32)
    for t in range(1, T):
        X[t] = coef * X[t - 1] + rng.standard_normal(V).astype(np.float32)
    return X


# ---------------------------------------------------------------------------
# Regresor white-box
# ---------------------------------------------------------------------------

class WhiteBoxRegressor(nn.Module):
    """
    f(X) = Σ_{(t,v) ∈ A} x_{t,v}² + ε,  ε ~ N(0, noise_std²)
    Diferenciable (PyTorch) para que IG pueda calcular gradientes; el término
    de ruido no depende de x, por lo que no contribuye al gradiente (∂ε/∂x=0)
    y solo introduce varianza en la señal observada por los métodos de
    perturbación, simulando una función objetivo estocástica.
    """

    def __init__(self, salient_mask: np.ndarray, noise_std: float = 0.0):
        """salient_mask: (T, V) bool array."""
        super().__init__()
        mask_t = torch.tensor(salient_mask, dtype=torch.float32)
        self.register_buffer("mask", mask_t)
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, V) → (N,)
        out = (x ** 2 * self.mask.unsqueeze(0)).sum(dim=(1, 2))
        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std
        return out


def make_predict_fn(model: WhiteBoxRegressor, device: str = "cpu"):
    """Envuelve el modelo PyTorch en una función predict_fn compatible."""
    model.eval()

    def fn(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).to(device)
            return model(t).cpu().numpy()

    return fn


# ---------------------------------------------------------------------------
# Evaluación de un método sobre un escenario
# ---------------------------------------------------------------------------

def evaluate_method(alphas: np.ndarray,
                    true_mask: np.ndarray) -> dict:
    """
    Promedia las métricas sobre las N explicaciones del lote.

    Parámetros
    ----------
    alphas    : (N, T, V)
    true_mask : (T, V) bool

    Retorna
    -------
    dict con valores promediados
    """
    results = [compute_all_metrics(alphas[i], true_mask) for i in range(len(alphas))]
    keys = list(results[0].keys())
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


# ---------------------------------------------------------------------------
# Experimento principal
# ---------------------------------------------------------------------------

def run_whitebox_experiments(
    T: int = 20,
    V: int = 15,
    N: int = 30,               # muestras de test por repetición
    n_rep: int = 10,           # repeticiones (= "10 times" en Dynamask)
    n_salient_features: int = 3,   # |A_X| para "rare feature"
    n_salient_times: int = 4,      # |A_T| para "rare time"
    n_svs_samples: int = 20,
    relative_noise: float = 0.0,   # σ_ε como fracción del std de la señal limpia f(X)
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Ejecuta los dos escenarios del Experimento 1 (rare feature + rare time).

    Retorna
    -------
    {
      "rare_feature": { método: {métrica: (mean, std)} },
      "rare_time":    { método: {métrica: (mean, std)} },
    }
    """
    scenarios = {
        "rare_feature": {
            "A_T_size": T,               # todos los tiempos son salientes
            "A_X_size": n_salient_features,
        },
        "rare_time": {
            "A_T_size": n_salient_times,  # pocos tiempos son salientes
            "A_X_size": V,               # todas las features son salientes
        },
    }

    method_names = ["FO", "FP", "IG", "SVS"]
    all_results = {}

    for scenario_name, cfg in scenarios.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Escenario: {scenario_name}  "
                  f"(|A_T|={cfg['A_T_size']}, |A_X|={cfg['A_X_size']})")
            print(f"{'='*60}")

        # acumula métricas por método: método → lista de dicts (una por rep)
        rep_results: dict[str, list] = {m: [] for m in method_names}

        for rep in range(n_rep):
            rng = np.random.default_rng(rep * 137 + 7)

            # ---- Ground truth mask ----
            salient_features = rng.choice(V, cfg["A_X_size"], replace=False)
            salient_times = rng.choice(T, cfg["A_T_size"], replace=False)
            true_mask = np.zeros((T, V), dtype=bool)
            true_mask[np.ix_(salient_times, salient_features)] = True

            # ---- Referencia = cero (el modelo es cuadrático, referencia natural) ----
            reference = np.zeros(V, dtype=np.float32)

            # ---- Generar datos de test ----
            X_test = np.stack([generate_ar1(T, V, rng) for _ in range(N)])

            # ---- Modelo white-box (limpio primero, para calibrar el ruido) ----
            clean_model = WhiteBoxRegressor(true_mask).to(device)
            if relative_noise > 0:
                with torch.no_grad():
                    f_clean = clean_model(torch.tensor(X_test, dtype=torch.float32)).numpy()
                noise_std = float(relative_noise * f_clean.std())
            else:
                noise_std = 0.0
            wb_model = WhiteBoxRegressor(true_mask, noise_std=noise_std).to(device)
            predict_fn = make_predict_fn(wb_model, device)

            if verbose:
                print(f"\n  Rep {rep+1}/{n_rep} — "
                      f"salient_features={sorted(salient_features.tolist())}, "
                      f"salient_times={sorted(salient_times.tolist())}, "
                      f"noise_std={noise_std:.3f}")

            # ---- FO (Feature Occlusion = método propuesto TPE) ----
            if verbose:
                print(f"    FO...", end=" ", flush=True)
            fo_engine = TemporalPerturbationEngine(predict_fn, reference)
            fo_alphas = fo_engine.explain_batch(X_test, normalize=True,
                                                verbose=False)
            rep_results["FO"].append(evaluate_method(fo_alphas, true_mask))
            if verbose:
                print("listo")

            # ---- FP (Feature Permutation) ----
            if verbose:
                print(f"    FP...", end=" ", flush=True)
            fp = FPExplainer(predict_fn, reference, X_background=X_test, seed=rep)
            fp_alphas = fp.explain_batch(X_test, normalize=True, verbose=False)
            rep_results["FP"].append(evaluate_method(fp_alphas, true_mask))
            if verbose:
                print("listo")

            # ---- IG (Integrated Gradients) ----
            if verbose:
                print(f"    IG...", end=" ", flush=True)
            ig = IGExplainer(wb_model, reference, device=device, n_steps=50)
            ig_alphas = ig.explain_batch(X_test, normalize=True, verbose=False)
            rep_results["IG"].append(evaluate_method(ig_alphas, true_mask))
            if verbose:
                print("listo")

            # ---- SVS (Shapley Value Sampling) ----
            if verbose:
                print(f"    SVS...", end=" ", flush=True)
            svs = SVSExplainer(predict_fn, reference,
                               n_samples=n_svs_samples, seed=rep)
            svs_alphas = svs.explain_batch(X_test[:5], normalize=True,
                                           verbose=False)  # 5 muestras para SVS
            # expandimos a N repitiendo para mantener el shape compatible
            # (en práctica el paper usa el mismo N para todos; aquí limitamos SVS)
            rep_results["SVS"].append(evaluate_method(svs_alphas, true_mask))
            if verbose:
                print("listo")

        # ---- Agrega por repetición: media ± std ----
        scenario_agg: dict[str, dict] = {}
        for method in method_names:
            keys = list(rep_results[method][0].keys())
            agg = {}
            for k in keys:
                vals = [rep_results[method][r][k] for r in range(n_rep)]
                agg[k] = {"mean": float(np.mean(vals)),
                           "std": float(np.std(vals))}
            scenario_agg[method] = agg

        all_results[scenario_name] = scenario_agg

    return all_results


def print_results_table(results: dict) -> None:
    """Imprime la tabla en formato análogo a las Tablas 1 y 2 de Dynamask."""
    metrics_show = ["AUP", "AUR", "IM_A", "SM_A"]

    for scenario, methods in results.items():
        print(f"\n{'='*70}")
        print(f"  Tabla: {scenario.upper().replace('_', ' ')}")
        print(f"{'='*70}")
        header = f"{'Método':<10}" + "".join(f"{m:>20}" for m in metrics_show)
        print(header)
        print("-" * 70)
        for method, mdict in methods.items():
            row = f"{method:<10}"
            for m in metrics_show:
                if m in mdict:
                    mean = mdict[m]["mean"]
                    std = mdict[m]["std"]
                    row += f"  {mean:.2f} ± {std:.2f}  ".rjust(20)
                else:
                    row += f"{'N/A':>20}"
            print(row)
        print()


if __name__ == "__main__":
    import time

    print("Ejecutando Experimento 1 — White-box Sintético")
    print("(análogo a Tablas 1 y 2 de Dynamask)\n")

    t0 = time.time()
    results = run_whitebox_experiments(
        T=20,
        V=15,
        N=20,
        n_rep=10,
        n_salient_features=3,
        n_salient_times=4,
        n_svs_samples=15,
        verbose=True,
    )
    elapsed = time.time() - t0

    print_results_table(results)
    print(f"\nTiempo total: {elapsed:.1f}s")

    # Guardar JSON
    out_path = Path(__file__).parent.parent / "results" / "exp1_whitebox_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Resultados guardados en: {out_path}")
