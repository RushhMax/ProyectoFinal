"""
Clase principal del framework de explicabilidad.

Orquesta la interacción entre modelos heterogéneos y el motor
de perturbaciones, y expone una API uniforme para todos los modelos.
"""
import numpy as np
import torch
import sys
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).parent.parent))
from framework.perturbation import TemporalPerturbationEngine
from config import FEATURES


class ModelAgnosticExplainer:
    """
    Parámetros
    ----------
    model : objeto con método .predict(X) o módulo PyTorch.
    model_type : 'pytorch' | 'sklearn'
    reference : np.ndarray (V,)
        Media del conjunto de entrenamiento por variable (escala normalizada).
    device : str, opcional — solo para modelos PyTorch.
    """

    def __init__(self,
                 model,
                 model_type: str,
                 reference: np.ndarray,
                 device: str = "cpu"):
        self.model = model
        self.model_type = model_type
        self.device = device
        self.engine = TemporalPerturbationEngine(
            predict_fn=self._predict_fn(),
            reference=reference,
        )

    def _predict_fn(self):
        if self.model_type == "pytorch":
            device = self.device
            model = self.model
            model.eval()

            def fn(X: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    t = torch.tensor(X, dtype=torch.float32).to(device)
                    return model(t).cpu().numpy()

            return fn

        elif self.model_type == "sklearn":
            return self.model.predict

        else:
            raise ValueError(f"model_type desconocido: {self.model_type}")

    def explain(self, x: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Explicación para una sola ventana (T, V)."""
        return self.engine.explain(x, normalize=normalize)

    def explain_batch(self,
                      X: np.ndarray,
                      normalize: bool = True,
                      verbose: bool = True) -> np.ndarray:
        """Explicaciones para un lote (N, T, V). Retorna (N, T, V)."""
        return self.engine.explain_batch(X, normalize=normalize, verbose=verbose)

    def mean_importance_by_variable(self, alphas: np.ndarray) -> np.ndarray:
        """Promedio de importancia por variable sobre N predicciones. Retorna (V,)."""
        return alphas.mean(axis=(0, 1))   # media sobre N y T

    def mean_importance_by_timestep(self, alphas: np.ndarray) -> np.ndarray:
        """Promedio de importancia por paso de tiempo. Retorna (T,)."""
        return alphas.mean(axis=(0, 2))   # media sobre N y V

    def mean_importance_matrix(self, alphas: np.ndarray) -> np.ndarray:
        """Matriz promedio T×V sobre todas las predicciones explicadas."""
        return alphas.mean(axis=0)        # (T, V)
