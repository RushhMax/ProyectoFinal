"""
Evaluación de consistencia entre explicaciones de distintos modelos.

Mide si los patrones de importancia temporal son similares entre
arquitecturas heterogéneas sobre las mismas predicciones.
"""
import numpy as np
from scipy.stats import spearmanr, kendalltau
from scipy.spatial.distance import cosine
from typing import Dict, Tuple


def spearman_correlation(alpha_a: np.ndarray, alpha_b: np.ndarray) -> float:
    """Correlación de Spearman entre dos matrices aplanadas."""
    rho, _ = spearmanr(alpha_a.ravel(), alpha_b.ravel())
    return float(rho)


def cosine_similarity(alpha_a: np.ndarray, alpha_b: np.ndarray) -> float:
    """Similitud coseno entre dos matrices aplanadas."""
    return 1.0 - cosine(alpha_a.ravel(), alpha_b.ravel())


def pairwise_consistency(alphas: Dict[str, np.ndarray],
                         metric: str = "spearman") -> Dict[Tuple[str, str], float]:
    """
    Calcula consistencia entre todos los pares de modelos.

    Parámetros
    ----------
    alphas : dict nombre_modelo → array (T, V) o (N, T, V)
    metric : 'spearman' | 'cosine'

    Retorna
    -------
    dict (modelo_a, modelo_b) → score
    """
    fn = spearman_correlation if metric == "spearman" else cosine_similarity
    names = list(alphas.keys())
    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            score = fn(alphas[a], alphas[b])
            results[(a, b)] = score
    return results


def mean_pairwise_consistency(alphas_batch: Dict[str, np.ndarray],
                               metric: str = "spearman") -> Dict[Tuple[str, str], float]:
    """
    Calcula consistencia media sobre un lote de N predicciones.

    alphas_batch : dict nombre → (N, T, V)
    """
    names = list(alphas_batch.keys())
    N = next(iter(alphas_batch.values())).shape[0]
    fn = spearman_correlation if metric == "spearman" else cosine_similarity

    pair_scores: Dict[Tuple[str, str], list] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pair_scores[(a, b)] = []

    for n in range(N):
        for (a, b) in pair_scores:
            score = fn(alphas_batch[a][n], alphas_batch[b][n])
            pair_scores[(a, b)].append(score)

    return {pair: float(np.mean(scores)) for pair, scores in pair_scores.items()}
