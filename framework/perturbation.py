"""
Motor de perturbaciones temporales.

Para cada predicción x (ventana T×V), calcula la matriz de importancia
α(t,v) = |f(x) - f(x_{-(t,v)})| perturbando una celda a la vez y
respetando el orden causal de la serie temporal.
"""
import numpy as np
from typing import Callable


class TemporalPerturbationEngine:
    """
    Parámetros
    ----------
    predict_fn : callable
        Función que acepta un array (N, T, V) y devuelve (N,).
        Debe funcionar con numpy arrays (el motor convierte modelos
        PyTorch externamente si es necesario).
    reference : np.ndarray, shape (V,)
        Valor de referencia para enmascarar cada celda.
        Por defecto se usa la media de entrenamiento por variable.
    """

    def __init__(self,
                 predict_fn: Callable[[np.ndarray], np.ndarray],
                 reference: np.ndarray):
        self.predict_fn = predict_fn
        self.reference = reference  # (V,)

    def explain(self, x: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Calcula la matriz de importancia para una sola ventana.

        Parámetros
        ----------
        x : np.ndarray, shape (T, V)
        normalize : bool
            Si True aplica softmax para que los valores sumen 1.

        Retorna
        -------
        alpha : np.ndarray, shape (T, V)
        """
        T, V = x.shape

        # Construye batch: [original, pert(0,0), pert(0,1), ..., pert(T-1,V-1)]
        # Una sola llamada a predict_fn en vez de T*V+1 llamadas individuales.
        batch = np.tile(x, (T * V + 1, 1, 1))  # (T*V+1, T, V)
        for idx, (t, v) in enumerate(np.ndindex(T, V)):
            batch[idx + 1, t, v] = self.reference[v]

        preds = self.predict_fn(batch)  # (T*V+1,)
        f_original = float(preds[0])
        alpha = np.abs(preds[1:].astype(np.float32) - f_original).reshape(T, V)

        if normalize:
            total = alpha.sum()
            if total > 0:
                alpha = alpha / total

        return alpha

    def explain_batch(self,
                      X: np.ndarray,
                      normalize: bool = True,
                      verbose: bool = True) -> np.ndarray:
        """
        Calcula matrices de importancia para un lote de ventanas.

        Parámetros
        ----------
        X : np.ndarray, shape (N, T, V)

        Retorna
        -------
        alphas : np.ndarray, shape (N, T, V)
        """
        N = X.shape[0]
        alphas = []
        for i, x in enumerate(X):
            if verbose and (i % 10 == 0 or i == N - 1):
                print(f"  Explicando {i+1}/{N}...", end="\r")
            alphas.append(self.explain(x, normalize=normalize))
        if verbose:
            print()
        return np.stack(alphas)  # (N, T, V)
