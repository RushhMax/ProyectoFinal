"""Preprocesamiento: escalado, ventanas deslizantes y splits train/test."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURES, TARGET, WINDOW_SIZE, HORIZON, TRAIN_RATIO, VAL_RATIO, DATA_DIR
from data.features import add_indicators


def split_temporal(df: pd.DataFrame,
                   train_ratio: float = TRAIN_RATIO
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el DataFrame en train/test respetando el orden temporal."""
    n = len(df)
    cut = int(n * train_ratio)
    return df.iloc[:cut], df.iloc[cut:]


def fit_scaler(train_df: pd.DataFrame,
               features: list = FEATURES,
               path: Path = DATA_DIR / "scaler.pkl") -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(train_df[features].values)
    joblib.dump(scaler, path)
    return scaler


def load_scaler(path: Path = DATA_DIR / "scaler.pkl") -> MinMaxScaler:
    return joblib.load(path)


def scale(df: pd.DataFrame,
          scaler: MinMaxScaler,
          features: list = FEATURES) -> np.ndarray:
    return scaler.transform(df[features].values)


def make_windows(data: np.ndarray,
                 window: int = WINDOW_SIZE,
                 horizon: int = HORIZON,
                 target_col: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye ventanas deslizantes X de forma (N, T, V) e y de forma (N,).

    target_col: índice de la columna objetivo en `data`.
                Si es None se infiere desde FEATURES y TARGET.
    """
    if target_col is None:
        target_col = FEATURES.index(TARGET)

    X, y = [], []
    for i in range(len(data) - window - horizon + 1):
        X.append(data[i : i + window])
        y.append(data[i + window + horizon - 1, target_col])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def inverse_target(values: np.ndarray,
                   scaler: MinMaxScaler,
                   target_col: int = None) -> np.ndarray:
    """Desnormaliza los valores del target al espacio original."""
    if target_col is None:
        target_col = FEATURES.index(TARGET)
    dummy = np.zeros((len(values), len(FEATURES)))
    dummy[:, target_col] = values
    return scaler.inverse_transform(dummy)[:, target_col]


def split_temporal_3way(df: pd.DataFrame,
                        train_ratio: float = TRAIN_RATIO,
                        val_ratio: float = VAL_RATIO,
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide en train / val / test respetando el orden temporal.

    train_ratio controla el corte train+val vs. test (igual que TRAIN_RATIO).
    val_ratio es la fracción del periodo train+val dedicada a validación.
    Resultado: train = train_ratio*(1-val_ratio), val = train_ratio*val_ratio,
               test  = 1 - train_ratio.
    """
    n = len(df)
    cut_test  = int(n * train_ratio)                       # mismo corte que split_temporal
    cut_train = int(cut_test * (1.0 - val_ratio))         # val al final del periodo train
    return df.iloc[:cut_train], df.iloc[cut_train:cut_test], df.iloc[cut_test:]


def prepare_data(raw_df: pd.DataFrame
                 ) -> Tuple[np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray,
                            MinMaxScaler]:
    """
    Pipeline completo: split → scaler → ventanas.
    Devuelve X_train, y_train, X_test, y_test, scaler.
    """
    df = add_indicators(raw_df)
    train_df, test_df = split_temporal(df)
    scaler = fit_scaler(train_df)

    train_scaled = scale(train_df, scaler)
    test_scaled = scale(test_df, scaler)

    X_train, y_train = make_windows(train_scaled)
    X_test, y_test = make_windows(test_scaled)

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test : X={X_test.shape}, y={y_test.shape}")
    return X_train, y_train, X_test, y_test, scaler


def prepare_data_with_val(raw_df: pd.DataFrame,
                          train_ratio: float = TRAIN_RATIO,
                          val_ratio: float = VAL_RATIO,
                          ) -> Tuple[np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray,
                                     MinMaxScaler]:
    """
    Pipeline con split temporal estricto y sin leakage en early stopping.

    Estrategia:
      - El scaler se ajusta sobre el periodo de desarrollo completo (train+val =
        TRAIN_RATIO del total), garantizando que cubre el rango de precios del
        período de entrenamiento sin filtrar información del test.
      - El periodo de desarrollo se divide temporalmente en train (90%) y val (10%)
        ANTES de construir ventanas, eliminando el leakage de ventanas solapadas.
      - El test (último 1-TRAIN_RATIO) permanece completamente separado.

    Devuelve X_train, y_train, X_val, y_val, X_test, y_test, scaler.
    """
    df = add_indicators(raw_df)
    dev_df, test_df = split_temporal(df, train_ratio)   # mismo corte que prepare_data

    # Scaler ajustado en todo el periodo de desarrollo (train + val)
    scaler = fit_scaler(dev_df)

    # Split temporal dentro del periodo de desarrollo (sin leakage)
    cut = int(len(dev_df) * (1.0 - val_ratio))
    train_df = dev_df.iloc[:cut]
    val_df   = dev_df.iloc[cut:]

    X_train, y_train = make_windows(scale(train_df, scaler))
    X_val,   y_val   = make_windows(scale(val_df,   scaler))
    X_test,  y_test  = make_windows(scale(test_df,  scaler))

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val  : X={X_val.shape},   y={y_val.shape}")
    print(f"Test : X={X_test.shape},  y={y_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


if __name__ == "__main__":
    from data.download import load_raw
    df = load_raw()
    X_tr, y_tr, X_te, y_te, sc = prepare_data(df)
