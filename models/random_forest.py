"""Modelo Random Forest (sklearn) envuelto con la misma interfaz de predicción."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import sys
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE, MODELS_DIR


class RandomForestModel:
    """Wrapper que acepta entradas (N, T, V) aplanándolas antes de llamar a sklearn."""

    def __init__(self,
                 n_estimators: int = RF_N_ESTIMATORS,
                 max_depth=RF_MAX_DEPTH,
                 random_state: int = RF_RANDOM_STATE):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        # (N, T, V) → (N, T*V)
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        self.model.fit(self._flatten(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._flatten(X)).astype(np.float32)

    def save(self, path: Path = MODELS_DIR / "rf.pkl") -> None:
        joblib.dump(self.model, path)
        print(f"RF guardado en {path}")

    @classmethod
    def load(cls, path: Path = MODELS_DIR / "rf.pkl") -> "RandomForestModel":
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        return obj


if __name__ == "__main__":
    rf = RandomForestModel()
    X = np.random.randn(100, 20, 5).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    rf.fit(X, y)
    print(rf.predict(X[:5]).shape)  # (5,)
