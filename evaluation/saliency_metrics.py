"""
Métricas de evaluación de saliency maps compatibles con Dynamask
(Crabbé & van der Schaar, 2021).

Métricas para experimentos sintéticos (ground truth conocido):
  - AUP  : Area Under Precision curve (↑ better)
  - AUR  : Area Under Recall curve (↑ better)
  - IM(A): Mask Information Content (↑ better)
  - SM(A): Mask Entropy in non-salient region (↓ better)
  - AUROC, AUPRC: métricas de clasificación binaria

Métricas de faithfulness para datos reales (ground truth desconocido):
  - mae_shift: |f(X) - f(X̃)| cuando se reemplazan top-α celdas (↑ better)
  - direction_consistency: fracción donde sign(f(X)) = sign(f(X̃)) (↓ better)
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def normalize_minmax(alpha: np.ndarray) -> np.ndarray:
    """Normaliza alpha a [0, 1] via min-max."""
    mn, mx = alpha.min(), alpha.max()
    if mx - mn < 1e-10:
        return np.full_like(alpha, 0.5)
    return (alpha - mn) / (mx - mn)


def aup(alpha: np.ndarray, true_mask: np.ndarray, n_thresh: int = 100) -> float:
    """
    Area Under Precision curve.
    Barre θ ∈ [0,1]; predice salient si m(t,v) ≥ θ; calcula precision.
    Retorna el área (higher is better).
    """
    m = normalize_minmax(alpha)
    thresholds = np.linspace(0.0, 1.0, n_thresh)
    precisions = []
    for theta in thresholds:
        pred = m >= theta
        tp = float((pred & true_mask).sum())
        pp = float(pred.sum())
        precisions.append(tp / pp if pp > 0 else 1.0)
    return float(np.trapezoid(precisions, thresholds))


def aur(alpha: np.ndarray, true_mask: np.ndarray, n_thresh: int = 100) -> float:
    """
    Area Under Recall curve.
    Retorna el área (higher is better).
    """
    m = normalize_minmax(alpha)
    thresholds = np.linspace(0.0, 1.0, n_thresh)
    recalls = []
    pos = float(true_mask.sum())
    for theta in thresholds:
        pred = m >= theta
        tp = float((pred & true_mask).sum())
        recalls.append(tp / pos if pos > 0 else 1.0)
    return float(np.trapezoid(recalls, thresholds))


def information_content(alpha: np.ndarray,
                        true_mask: np.ndarray,
                        eps: float = 1e-8) -> float:
    """
    IM(A) = -Σ_{(t,v) ∈ A} log(1 - m_{t,v}).
    Higher is better (mask concentra peso en región saliente).
    """
    m = normalize_minmax(alpha)
    return float(-np.sum(np.log(1.0 - m[true_mask.astype(bool)] + eps)))


def mask_entropy(alpha: np.ndarray,
                 true_mask: np.ndarray,
                 eps: float = 1e-8) -> float:
    """
    SM(A) = Σ_{(t,v) ∉ A} H_bin(m_{t,v}).
    Lower is better (mask es nítida fuera de la región saliente).
    """
    m = normalize_minmax(alpha)
    non_salient = ~true_mask.astype(bool)
    p = m[non_salient].clip(eps, 1.0 - eps)
    return float(-np.sum(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


def auroc_auprc(alpha: np.ndarray,
                true_mask: np.ndarray):
    """
    Trata la importancia como score de clasificación binaria.
    Retorna (AUROC, AUPRC).
    """
    y_score = normalize_minmax(alpha).ravel()
    y_true = true_mask.ravel().astype(int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan"), float("nan")
    return (float(roc_auc_score(y_true, y_score)),
            float(average_precision_score(y_true, y_score)))


def compute_all_metrics(alpha: np.ndarray,
                        true_mask: np.ndarray) -> dict:
    """
    Calcula AUP, AUR, IM(A), SM(A), AUROC, AUPRC para un par (alpha, true_mask).
    """
    a_roc, a_prc = auroc_auprc(alpha, true_mask)
    return {
        "AUP": aup(alpha, true_mask),
        "AUR": aur(alpha, true_mask),
        "IM_A": information_content(alpha, true_mask),
        "SM_A": mask_entropy(alpha, true_mask),
        "AUROC": a_roc,
        "AUPRC": a_prc,
    }


# ---------------------------------------------------------------------------
# Métricas de faithfulness para datos reales (Experimento 3 / Exp. 3 analog)
# ---------------------------------------------------------------------------

def faithfulness_shift(predict_fn,
                       X: np.ndarray,
                       alphas: np.ndarray,
                       reference: np.ndarray,
                       alpha_frac: float = 0.1) -> float:
    """
    MAE shift al reemplazar la fracción alpha_frac de celdas más importantes
    con el valor de referencia.  Equivale al CE del paper Dynamask para
    regresión (higher is better).

    Parámetros
    ----------
    predict_fn : callable (N, T, V) → (N,)
    X          : (N, T, V) muestras test
    alphas     : (N, T, V) importancias
    reference  : (V,) baseline por variable
    alpha_frac : fracción de celdas a reemplazar

    Retorna
    -------
    float : E[|f(X) - f(X̃)|]
    """
    N, T, V = X.shape
    n_cells = max(1, int(alpha_frac * T * V))

    f_orig = predict_fn(X)

    X_pert = X.copy()
    for i in range(N):
        top_idx = np.argpartition(alphas[i].ravel(), -n_cells)[-n_cells:]
        t_idx, v_idx = np.unravel_index(top_idx, (T, V))
        X_pert[i, t_idx, v_idx] = reference[v_idx]

    f_pert = predict_fn(X_pert)
    return float(np.mean(np.abs(f_orig - f_pert)))


def direction_consistency(predict_fn,
                          X: np.ndarray,
                          alphas: np.ndarray,
                          reference: np.ndarray,
                          alpha_frac: float = 0.1) -> float:
    """
    Fracción de predicciones cuyo signo NO cambia tras perturbar top-α celdas.
    Análogo al ACC del paper Dynamask (lower is better = más predicciones
    cambian de dirección, indicando que las celdas eran realmente importantes).

    Las predicciones se comparan contra la media del conjunto (predicción
    neutral), usando sign(f(X) - mean_pred).
    """
    N, T, V = X.shape
    n_cells = max(1, int(alpha_frac * T * V))

    f_orig = predict_fn(X)
    mean_pred = float(f_orig.mean())

    X_pert = X.copy()
    for i in range(N):
        top_idx = np.argpartition(alphas[i].ravel(), -n_cells)[-n_cells:]
        t_idx, v_idx = np.unravel_index(top_idx, (T, V))
        X_pert[i, t_idx, v_idx] = reference[v_idx]

    f_pert = predict_fn(X_pert)
    same_sign = np.sign(f_orig - mean_pred) == np.sign(f_pert - mean_pred)
    return float(same_sign.mean())


def faithfulness_curve(predict_fn,
                       X: np.ndarray,
                       alphas: np.ndarray,
                       reference: np.ndarray,
                       fracs: np.ndarray = None) -> dict:
    """
    Curva de faithfulness (MAE shift y direction consistency) para varios α.
    Retorna dict con claves 'fracs', 'mae_shift', 'dir_consistency'.
    """
    if fracs is None:
        fracs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    shifts = []
    dircons = []
    for frac in fracs:
        shifts.append(faithfulness_shift(predict_fn, X, alphas, reference, frac))
        dircons.append(direction_consistency(predict_fn, X, alphas, reference, frac))

    return {
        "fracs": fracs.tolist(),
        "mae_shift": shifts,
        "dir_consistency": dircons,
    }
