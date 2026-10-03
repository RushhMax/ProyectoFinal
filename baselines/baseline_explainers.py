"""
Métodos de explicabilidad baseline para comparación con Dynamask
(Crabbé & van der Schaar, 2021).

Todos producen una matriz de importancia (T, V) o (N, T, V) compatible
con el motor de perturbaciones del framework propuesto.

Métodos implementados
---------------------
FPExplainer  : Feature Permutation — reemplaza (t,v) con valor de otra muestra
IGExplainer  : Integrated Gradients — sólo modelos PyTorch diferenciables
SVSExplainer : Shapley Value Sampling — aproximación por permutaciones aleatorias
"""
import numpy as np
import torch
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Feature Permutation (FP)
# ---------------------------------------------------------------------------

class FPExplainer:
    """
    Feature Permutation: la importancia de (t,v) se mide como el cambio
    en la predicción al reemplazar x_{t,v} con un valor muestreado de la
    distribución marginal de esa variable (tomado de X_background).

    Si X_background es None, usa el vector de referencia (equivale a FO).
    """

    def __init__(self,
                 predict_fn: Callable[[np.ndarray], np.ndarray],
                 reference: np.ndarray,
                 X_background: Optional[np.ndarray] = None,
                 seed: int = 42):
        self.predict_fn = predict_fn
        self.reference = np.asarray(reference, dtype=np.float32)
        self.X_background = X_background   # (N_bg, T, V) o (N_bg, V)
        self.rng = np.random.default_rng(seed)

    def explain(self, x: np.ndarray, normalize: bool = True) -> np.ndarray:
        """x: (T, V) → alpha: (T, V)."""
        T, V = x.shape
        n_cells = T * V

        # batch[0] = original; batch[k+1] = perturba celda k
        batch = np.tile(x[np.newaxis], (n_cells + 1, 1, 1))

        for idx, (t, v) in enumerate(np.ndindex(T, V)):
            if self.X_background is not None:
                bg_idx = int(self.rng.integers(len(self.X_background)))
                bg = self.X_background[bg_idx]
                # bg puede ser (T, V) o (V,)
                val = float(bg[t, v] if bg.ndim == 2 else bg[v])
            else:
                val = float(self.reference[v])
            batch[idx + 1, t, v] = val

        preds = self.predict_fn(batch)
        alpha = np.abs(preds[1:].astype(np.float32) - float(preds[0])).reshape(T, V)

        if normalize:
            s = alpha.sum()
            if s > 0:
                alpha = alpha / s
        return alpha

    def explain_batch(self,
                      X: np.ndarray,
                      normalize: bool = True,
                      verbose: bool = True) -> np.ndarray:
        """X: (N, T, V) → alphas: (N, T, V)."""
        N = len(X)
        out = []
        for i, x in enumerate(X):
            if verbose and (i % 10 == 0 or i == N - 1):
                print(f"  FP {i+1}/{N}...", end="\r")
            out.append(self.explain(x, normalize))
        if verbose:
            print()
        return np.stack(out)


# ---------------------------------------------------------------------------
# Integrated Gradients (IG)
# ---------------------------------------------------------------------------

class IGExplainer:
    """
    Integrated Gradients (Sundararajan et al., 2017) para modelos PyTorch.

    IG(t,v) = (x_{t,v} - ref_{t,v}) * ∫_0^1 ∂f/∂x_{t,v}(ref + α(x-ref)) dα

    La integral se aproxima con n_steps trapecios (regla de Riemann izquierda).
    """

    def __init__(self,
                 model: torch.nn.Module,
                 reference: np.ndarray,
                 device: str = "cpu",
                 n_steps: int = 50):
        self.model = model
        self.reference = np.asarray(reference, dtype=np.float32)   # (V,)
        self.device = device
        self.n_steps = n_steps
        self.model.eval()

    def explain(self, x: np.ndarray, normalize: bool = True) -> np.ndarray:
        """x: (T, V) → alpha: (T, V)."""
        T, V = x.shape
        x_t = torch.tensor(x, dtype=torch.float32)
        # referencia replicada a (T, V)
        ref_t = torch.tensor(
            np.tile(self.reference, (T, 1)), dtype=torch.float32
        )
        delta = x_t - ref_t   # (T, V)

        ig = torch.zeros(T, V)
        self.model.eval()

        for k in range(self.n_steps):
            alpha_k = k / self.n_steps
            interp = (ref_t + alpha_k * delta).unsqueeze(0).to(self.device)
            interp = interp.detach().requires_grad_(True)

            out = self.model(interp)
            out.sum().backward()

            if interp.grad is not None:
                ig += interp.grad.squeeze(0).cpu().detach()

        ig = ig / self.n_steps
        attr = (delta * ig).abs().numpy()

        if normalize:
            s = attr.sum()
            if s > 0:
                attr = attr / s
        return attr

    def explain_batch(self,
                      X: np.ndarray,
                      normalize: bool = True,
                      verbose: bool = True) -> np.ndarray:
        """X: (N, T, V) → alphas: (N, T, V)."""
        N = len(X)
        out = []
        for i, x in enumerate(X):
            if verbose and (i % 10 == 0 or i == N - 1):
                print(f"  IG {i+1}/{N}...", end="\r")
            out.append(self.explain(x, normalize))
        if verbose:
            print()
        return np.stack(out)


# ---------------------------------------------------------------------------
# Shapley Value Sampling (SVS)
# ---------------------------------------------------------------------------

class SVSExplainer:
    """
    Shapley Value Sampling (Castro et al., 2009).

    Aproxima el valor de Shapley de cada celda (t,v) muestreando
    n_samples permutaciones aleatorias de todas las celdas.

    Para cada permutación σ y posición donde aparece la celda objetivo,
    se mide la contribución marginal:
        v_{t,v} ≈ E_σ[ |f(S_σ ∪ {(t,v)}) - f(S_σ)| ]

    Internamente vectoriza: evalúa 1 batch de 2 muestras por celda
    en cada permutación, reutilizando la evaluación acumulada.
    """

    def __init__(self,
                 predict_fn: Callable[[np.ndarray], np.ndarray],
                 reference: np.ndarray,
                 n_samples: int = 20,
                 seed: int = 42):
        self.predict_fn = predict_fn
        self.reference = np.asarray(reference, dtype=np.float32)   # (V,)
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def explain(self, x: np.ndarray, normalize: bool = True) -> np.ndarray:
        """x: (T, V) → alpha: (T, V)."""
        T, V = x.shape
        n_cells = T * V
        svs = np.zeros(n_cells)

        # referencia como input (T, V) de respaldo
        x_ref = np.tile(self.reference, (T, 1)).astype(np.float32)

        for _ in range(self.n_samples):
            perm = self.rng.permutation(n_cells)   # orden de celdas en esta permutación
            x_S = x_ref.copy()  # coalición acumulada; arranca con todo en referencia

            # construimos un batch grande: 2 inputs por celda
            # [x_S antes de añadir celda k, x_S después de añadir celda k]
            batch_before = np.zeros((n_cells, T, V), dtype=np.float32)
            batch_after = np.zeros((n_cells, T, V), dtype=np.float32)

            for pos, cell_idx in enumerate(perm):
                t_c, v_c = divmod(int(cell_idx), V)
                batch_before[pos] = x_S.copy()
                x_S[t_c, v_c] = x[t_c, v_c]    # añade celda a coalición
                batch_after[pos] = x_S.copy()

            # evalúa las 2*n_cells muestras de una vez
            combined = np.concatenate([batch_before, batch_after], axis=0)
            preds = self.predict_fn(combined)
            diff = np.abs(preds[n_cells:] - preds[:n_cells])

            # acumula en el índice de celda correspondiente
            for pos, cell_idx in enumerate(perm):
                svs[cell_idx] += float(diff[pos])

        svs /= self.n_samples
        alpha = svs.reshape(T, V)

        if normalize:
            s = alpha.sum()
            if s > 0:
                alpha = alpha / s
        return alpha

    def explain_batch(self,
                      X: np.ndarray,
                      normalize: bool = True,
                      verbose: bool = True) -> np.ndarray:
        """X: (N, T, V) → alphas: (N, T, V)."""
        N = len(X)
        out = []
        for i, x in enumerate(X):
            if verbose and (i % 10 == 0 or i == N - 1):
                print(f"  SVS {i+1}/{N}...", end="\r")
            out.append(self.explain(x, normalize))
        if verbose:
            print()
        return np.stack(out)
