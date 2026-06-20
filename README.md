# Framework XAI Temporal para Predicción del S&P 500

Trabajo de fin de carrera (PFC) — Universidad Nacional de San Agustín.

Framework de explicabilidad agnóstico al modelo para series temporales financieras. Propone un motor de perturbaciones temporales que calcula la importancia de cada variable en cada paso de tiempo respetando causalidad, evaluado sobre tres arquitecturas heterogéneas: LSTM, Transformer y Random Forest.

---

## Estructura del proyecto

```
PFC/
├── config.py                  # Hiperparámetros y rutas centralizados
├── train.py                   # Entrena los 3 modelos
├── explain.py                 # Genera explicaciones, métricas y gráficos
│
├── data/
│   ├── download.py            # Descarga ^GSPC desde Yahoo Finance
│   ├── preprocessing.py       # Escalado, ventanas deslizantes, split temporal
│   ├── features.py            # Indicadores técnicos (SMA, MACD, RSI, ATR…)
│   └── sp500_raw.csv          # Datos históricos 2010–2024
│
├── models/
│   ├── lstm.py                # LSTM con mecanismo de atención
│   ├── transformer.py         # Transformer encoder
│   └── random_forest.py       # Random Forest (sklearn)
│
├── framework/
│   ├── perturbation.py        # Motor de perturbaciones temporales
│   └── explainer.py           # API uniforme para los 3 modelos
│
├── evaluation/
│   ├── consistency.py         # Consistencia inter-modelo (Spearman)
│   └── significance.py        # Significancia estadística (Mann-Whitney)
│
├── visualization/
│   └── plots.py               # Heatmaps, importancia por variable y por tiempo
│
├── saved_models/              # Checkpoints entrenados
│   ├── lstm_best.pt
│   └── transformer_best.pt
│
└── results/                   # Salidas generadas por explain.py
    ├── alphas_*.npy
    ├── metrics.json
    ├── explainability_results.json
    └── *.png
```

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Uso

### 1. Datos

El CSV ya está incluido. Si necesitas refrescarlo:

```bash
python data/download.py
```

### 2. Entrenamiento

```bash
python train.py
```

Entrena LSTM, Transformer y Random Forest. Los checkpoints se guardan en `saved_models/`.
> El modelo Random Forest (`rf.pkl`, ~52 MB) no está en el repositorio por tamaño. Se genera al correr `train.py`.

### 3. Explicaciones y evaluación

```bash
python explain.py
```

Genera las matrices de importancia α(t,v) para 100 muestras del test set, calcula consistencia entre modelos y significancia estadística, y produce los gráficos en `results/`.

---

## Resultados

Evaluado sobre el S&P 500 (2020–2024, precio medio ≈ 4,604 pts):

| Modelo | Accuracy (1−MAPE) | MAPE | R² | MAE |
|---|---|---|---|---|
| **LSTM** | **95.71%** | 4.29% | 0.876 | ±191 pts |
| Transformer | 95.51% | 4.49% | 0.793 | ±225 pts |
| Random Forest | 94.13% | 5.87% | 0.465 | ±308 pts |

### Consistencia XAI entre modelos (Spearman ρ)

| Par | ρ |
|---|---|
| LSTM vs Transformer | 0.834 |
| LSTM vs Random Forest | 0.680 |
| Transformer vs Random Forest | 0.604 |

Los tres modelos identifican patrones de importancia temporal estadísticamente significativos (p ≈ 10⁻³⁸ vs. explicaciones aleatorias, test Mann-Whitney).

---

## Metodología XAI

Para cada ventana de predicción **x** de forma (T=20, V=15):

$$\alpha(t, v) = \left| f(x) - f(x_{-(t,v)}) \right|$$

donde x_{-(t,v)} reemplaza la celda (t, v) con la media de entrenamiento de esa variable. La matriz resultante indica qué día y qué variable tienen mayor impacto causal en la predicción.

**Features utilizadas (15):** Open, High, Low, Close, Volume, SMA-5/10/20, MACD, MACD Signal, RSI-14, Bollinger Band width, ATR-14, Log Return, Volume ratio.

---

## Requisitos

- Python 3.10+
- PyTorch ≥ 2.0
- scikit-learn ≥ 1.3
- Ver `requirements.txt` para la lista completa
- 
