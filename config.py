from pathlib import Path

# Rutas
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "saved_models"
RESULTS_DIR = ROOT / "results"

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

# Datos
TICKER = "^GSPC"
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
OHLCV_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
FEATURES = [
    "Open", "High", "Low", "Close", "Volume",   # precio y volumen base
    "SMA_5", "SMA_10", "SMA_20",                 # tendencia
    "MACD", "MACD_Signal",                        # momentum
    "RSI_14",                                     # oscilador
    "BB_width",                                   # volatilidad relativa
    "ATR_14",                                     # volatilidad absoluta
    "Log_Return",                                 # retorno logarítmico
    "Volume_ratio",                               # volumen relativo
]
TARGET = "Close"
WINDOW_SIZE = 20       # T: pasos de tiempo por ventana (días)
HORIZON = 1            # pasos hacia adelante a predecir
TRAIN_RATIO = 0.8

# Entrenamiento
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
PATIENCE = 15          # early stopping

# LSTM
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2

# Transformer
TF_D_MODEL = 64
TF_NHEAD = 4
TF_LAYERS = 2
TF_DROPOUT = 0.1
TF_DIM_FF = 256

# Random Forest
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_RANDOM_STATE = 42

# Framework XAI
N_EXPLAIN = 100        # predicciones a explicar
RANDOM_STATE = 42
