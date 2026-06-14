"""
Evaluación de significancia estadística.

Verifica que las explicaciones generadas son estadísticamente distintas
de explicaciones producidas por azar (matrices de importancia aleatorias),
descartando que los patrones observados sean ruido.
"""
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon
from typing import Dict, Tuple


def random_baseline(shape: Tuple[int, ...],
                    n_samples: int = 1000,
                    seed: int = 42) -> np.ndarray:
    """
    Genera explicaciones aleatorias de referencia aplicando softmax sobre
    matrices uniformes, lo que mantiene la misma escala que las explicaciones
    reales normalizadas.
    """
    rng = np.random.default_rng(seed)
    from scipy.special import softmax
    samples = []
    for _ in range(n_samples):
        raw = rng.uniform(0, 1, shape)
        samples.append(softmax(raw.ravel()).reshape(shape))
    return np.stack(samples)  # (n_samples, T, V)


def significance_test(alphas: np.ndarray,
                      n_random: int = 1000,
                      seed: int = 42,
                      test: str = "mannwhitney"
                      ) -> Dict[str, float]:
    """
    Compara la distribución de valores de importancia real contra aleatoria.

    Parámetros
    ----------
    alphas : (N, T, V) — explicaciones reales
    test : 'mannwhitney' | 'ttest' | 'wilcoxon'

    Retorna
    -------
    dict con 'statistic' y 'p_value'
    """
    T, V = alphas.shape[1], alphas.shape[2]
    random_alphas = random_baseline((T, V), n_samples=n_random, seed=seed)

    real_flat = alphas.ravel()
    rand_flat = random_alphas.ravel()

    if test == "mannwhitney":
        stat, p = mannwhitneyu(real_flat, rand_flat, alternative="two-sided")
    elif test == "ttest":
        stat, p = ttest_ind(real_flat, rand_flat)
    elif test == "wilcoxon":
        # requiere misma longitud; muestreamos
        n = min(len(real_flat), len(rand_flat), 5000)
        rng = np.random.default_rng(seed)
        r = real_flat[rng.choice(len(real_flat), n, replace=False)]
        rd = rand_flat[rng.choice(len(rand_flat), n, replace=False)]
        stat, p = wilcoxon(r, rd)
    else:
        raise ValueError(f"test desconocido: {test}")

    return {"statistic": float(stat), "p_value": float(p)}


def per_model_significance(alphas_dict: Dict[str, np.ndarray],
                            n_random: int = 1000,
                            seed: int = 42,
                            test: str = "mannwhitney"
                            ) -> Dict[str, Dict[str, float]]:
    """Aplica la prueba de significancia para cada modelo."""
    return {
        name: significance_test(alphas, n_random, seed, test)
        for name, alphas in alphas_dict.items()
    }
