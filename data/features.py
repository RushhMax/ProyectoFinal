"""Indicadores técnicos sobre OHLCV del S&P 500."""
import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame OHLCV y devuelve uno enriquecido con indicadores técnicos.
    Se aplica sobre la serie completa (los rolling solo miran hacia atrás → sin leakage).
    Las filas con NaN iniciales se descartan con dropna().
    """
    out = df.copy()

    # Tendencia
    out["SMA_5"]  = out["Close"].rolling(5).mean()
    out["SMA_10"] = out["Close"].rolling(10).mean()
    out["SMA_20"] = out["Close"].rolling(20).mean()

    # Momentum — MACD
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"]        = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # Oscilador
    out["RSI_14"] = _rsi(out["Close"], 14)

    # Volatilidad relativa — Bollinger Band width normalizado
    sma20 = out["Close"].rolling(20).mean()
    std20 = out["Close"].rolling(20).std()
    out["BB_width"] = (2 * std20) / sma20

    # Volatilidad absoluta — ATR
    out["ATR_14"] = _atr(out, 14)

    # Retorno logarítmico diario
    out["Log_Return"] = np.log(out["Close"] / out["Close"].shift(1))

    # Volumen relativo (evita escala absoluta del volumen)
    vol_ma5 = out["Volume"].rolling(5).mean()
    out["Volume_ratio"] = out["Volume"] / vol_ma5

    return out.dropna()
