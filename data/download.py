"""Descarga y guarda los datos históricos del S&P 500 desde Yahoo Finance."""
import yfinance as yf
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TICKER, START_DATE, END_DATE, OHLCV_FEATURES, DATA_DIR


def download_sp500(ticker: str = TICKER,
                   start: str = START_DATE,
                   end: str = END_DATE) -> pd.DataFrame:
    print(f"Descargando {ticker} desde {start} hasta {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance >=0.2 devuelve MultiIndex (Price, Ticker); aplanamos al nivel Price
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[OHLCV_FEATURES].dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    print(f"  {len(df)} filas descargadas ({df.index[0].date()} – {df.index[-1].date()})")
    return df


def save_raw(df: pd.DataFrame, path: Path = DATA_DIR / "sp500_raw.csv") -> None:
    # Guarda con cabecera simple: Date,Open,High,Low,Close,Volume
    df.to_csv(path, index_label="Date")
    print(f"  Guardado en {path}")


def load_raw(path: Path = DATA_DIR / "sp500_raw.csv") -> pd.DataFrame:
    # Detecta si el CSV tiene el formato multi-fila que genera yfinance directamente
    with open(path) as f:
        first_line = f.readline().strip()

    if first_line.startswith("Price"):
        # Formato multi-fila: fila 0=Price/cols, fila 1=Ticker, fila 2=Date label
        df = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True)
        df.index.name = "Date"
    else:
        df = pd.read_csv(path, index_col="Date", parse_dates=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[OHLCV_FEATURES]


if __name__ == "__main__":
    df = download_sp500()
    save_raw(df)
    print(df.tail())
