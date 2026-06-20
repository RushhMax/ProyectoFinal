# Framework XAI Temporal para Predicción del S&P 500

Trabajo de fin de carrera (PFC) — Universidad Nacional de San Agustín de Arequipa.

Framework de **explicabilidad agnóstico al modelo** para series temporales financieras. Propone un motor de perturbaciones temporales que calcula la importancia causal de cada variable en cada paso de tiempo, evaluado sobre tres arquitecturas heterogéneas: LSTM, Transformer y Random Forest sobre el índice S&P 500 (2010–2024).

---

## Estructura del proyecto

```
PFC/
├── config.py                  # Hiperparámetros y rutas centralizados
├── train.py                   # Entrena los 3 modelos
├── explain.py                 # Genera explicaciones, métricas y gráficos
│
├── data/
│   ├── download.py            # Descarga ^GSPC desde Yahoo Finance (yfinance)
│   ├── preprocessing.py       # Escalado, ventanas deslizantes, split temporal
│   ├── features.py            # Indicadores técnicos (SMA, MACD, RSI, ATR…)
│   └── sp500_raw.csv          # Datos históricos 2010–2024 (incluido)
│
├── models/
│   ├── lstm.py                # LSTM 2 capas con mecanismo de atención temporal
│   ├── transformer.py         # Transformer encoder con attention pooling
│   └── random_forest.py       # Random Forest sklearn con interfaz uniforme
│
├── framework/
│   ├── perturbation.py        # Motor de perturbaciones: α(t,v) por celda
│   └── explainer.py           # API uniforme para PyTorch y sklearn
│
├── evaluation/
│   ├── consistency.py         # Consistencia inter-modelo (Spearman / cosine)
│   └── significance.py        # Significancia estadística (Mann-Whitney vs. azar)
│
├── visualization/
│   └── plots.py               # Heatmaps, importancia por variable y por tiempo
│
├── saved_models/              # Checkpoints entrenados
│   ├── lstm_best.pt           # Mejor LSTM según val loss
│   ├── transformer_best.pt    # Mejor Transformer según val loss
│   └── rf.pkl                 # Random Forest serializado
│
└── results/                   # Salidas generadas por explain.py
    ├── alphas_LSTM.npy            # Matrices α(t,v) — shape (100, 20, 15)
    ├── alphas_Transformer.npy
    ├── alphas_RandomForest.npy
    ├── metrics.json               # MSE, MAE, RMSE de los 3 modelos
    ├── explainability_results.json # Consistencia y significancia
    ├── heatmaps_comparison.png
    ├── variable_importance.png
    └── temporal_importance.png
```

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Requisitos principales:** Python 3.10+, PyTorch ≥ 2.0, scikit-learn ≥ 1.3, yfinance, seaborn, scipy.

---

## Uso

### 1. (Opcional) Refrescar datos

El CSV ya está incluido. Para descargar datos actualizados:

```bash
python data/download.py
```

### 2. Entrenamiento

```bash
python train.py
```

Entrena LSTM, Transformer y Random Forest con split temporal estricto (72% train / 8% val / 20% test). Los mejores checkpoints se guardan en `saved_models/`.

> El modelo Random Forest (`rf.pkl`, ~52 MB) no está en el repositorio por tamaño. Se genera al correr `train.py`.

### 3. Explicaciones y evaluación

```bash
python explain.py
```

Genera las matrices de importancia α(t,v) para 100 muestras del test set, calcula consistencia inter-modelo y significancia estadística, y produce los gráficos en `results/`.

---

## Metodología

### 1. Datos y features

Se utiliza el precio histórico del S&P 500 (`^GSPC`) desde 2010-01-01 hasta 2024-12-31, descargado vía `yfinance`. Sobre los datos OHLCV base se calculan **10 indicadores técnicos** sin forward leakage (todos usan solo datos históricos):

| Grupo | Features |
|---|---|
| Precio base | Open, High, Low, Close, Volume |
| Tendencia | SMA-5, SMA-10, SMA-20 |
| Momentum | MACD, MACD Signal |
| Oscilador | RSI-14 |
| Volatilidad relativa | Bollinger Band width |
| Volatilidad absoluta | ATR-14 |
| Retorno | Log Return diario |
| Volumen relativo | Volume ratio (Vol / SMA-5 Vol) |

**Total: 15 features (V=15), ventana temporal T=20 días.**

### 2. Split temporal y preprocesamiento

Para garantizar validez estadística en series temporales financieras se aplica una estrategia de split en tres etapas:

```
Datos completos (3 754 días, 2010–2024)
│
├── Período de desarrollo — 80% (2010–2022)
│   ├── MinMaxScaler ajustado sobre ESTE período completo
│   ├── Train — 72% del total (~2682 ventanas)  ← early stopping sin leakage
│   └── Val   —  8% del total (~281 ventanas)   ← separado antes de make_windows
│
└── Test — 20% del total (~731 ventanas, 2022–2024)
```

**Decisión clave:** el scaler se ajusta sobre el período de desarrollo completo (80%) y el split train/val se realiza **antes** de construir las ventanas deslizantes. Esto elimina el leakage que ocurría cuando las primeras ventanas de validación compartían historia con las últimas ventanas de entrenamiento.

**Por qué MinMaxScaler sobre el 80%:** el S&P 500 tiene tendencia de largo plazo (precio 2010 ≈ 1 000 pts, precio 2024 ≈ 5 000 pts). Ajustar el scaler solo sobre el 70% hace que los precios del período de validación y test queden completamente fuera del rango [0, 1], invalidando los modelos. Ajustar sobre el 80% (hasta finales de 2022, máximo histórico ≈ 4 790 pts) cubre la mayor parte del rango del test.

### 3. Arquitecturas de predicción

#### LSTM con atención temporal

Red LSTM de 2 capas (hidden=128, dropout=0.2) seguida de un mecanismo de atención aprendido que pondera cada paso de tiempo antes de la capa de salida:

```
x (batch, T, V) → LSTM → out (batch, T, 128)
                 → atención softmax → pesos (batch, T, 1)
                 → suma ponderada → contexto (batch, 128)
                 → FC → ŷ (batch,)
```

#### Transformer con attention pooling

Transformer encoder estándar (d_model=64, 4 heads, 2 capas, FFN=256) con positional encoding sinusoidal. La representación final se obtiene mediante **attention pooling aprendido** sobre todos los pasos de tiempo:

```
x (batch, T, V) → proyección lineal → pos. encoding
                → TransformerEncoder → (batch, T, 64)
                → atención softmax → pesos (batch, T, 1)
                → suma ponderada → (batch, 64) → FC → ŷ
```

> **Decisión arquitectónica importante:** se descartó el pooling por último token (`x[:, -1]`), que es la implementación naive pero induce artificialmente a que todas las explicaciones XAI se concentren en t-0 independientemente del modelo. El attention pooling permitió que el Transformer redujera su RMSE de 0.081 a 0.053.

#### Random Forest

Ensemble de 200 árboles (scikit-learn) que recibe las ventanas aplanadas (T×V = 300 features). Incluido como baseline de modelo tabular no-secuencial.

### 4. Framework XAI: motor de perturbaciones temporales

Para cada ventana de predicción **x** de forma (T=20, V=15), la importancia de la celda (t, v) se define como:

$$\alpha(t, v) = \left| f(x) - f(x_{-(t,v)}) \right|$$

donde $x_{-(t,v)}$ es la ventana original con la celda (t, v) reemplazada por la media de entrenamiento de esa variable (valor de referencia). La perturbación se realiza respetando la causalidad temporal: solo se altera una celda por vez.

**Implementación eficiente:** en lugar de T×V+1 llamadas al modelo, se construye un batch único de forma (T×V+1, T, V) con la ventana original y todas las perturbaciones, y se realiza **una sola llamada a predict**. Esto reduce el overhead de la capa de PyTorch/sklearn en un factor T×V.

La matriz resultante se normaliza por L1 (suma = 1) para hacer los valores comparables entre muestras. El **explainer** ofrece una API uniforme que adapta automáticamente modelos PyTorch y sklearn a la misma interfaz de `predict_fn`.

### 5. Evaluación del framework

**Consistencia inter-modelo (Spearman ρ):** mide si dos modelos distintos priorizan las mismas celdas (t, v) en sus explicaciones. Se calcula sobre los rankings de las matrices α promedio.

**Significancia estadística (Mann-Whitney):** contrasta la distribución de valores α reales contra un baseline aleatorio (matrices uniformes normalizadas con softmax). Una p < 0.05 indica que el modelo captura patrones genuinos y no ruido.

---

## Resultados

### Métricas de predicción (espacio normalizado MinMaxScaler)

| Modelo | MSE | MAE | RMSE |
|---|---|---|---|
| **LSTM** | 0.00342 | 0.0468 | **0.0585** |
| **Transformer** | **0.00284** | **0.0434** | **0.0533** |
| Random Forest | 0.1213 | 0.300 | 0.348 |

El Transformer supera al LSTM con attention pooling. El Random Forest presenta un RMSE elevado debido a su incapacidad de extrapolación: el 32.7% del test set (período 2023–2024) corresponde a precios que superan el máximo histórico del período de entrenamiento (~4 790 pts). Los árboles de decisión truncan sus predicciones al máximo observado durante el entrenamiento, lo que introduce un sesgo sistemático en el período alcista de 2023–2024. Este comportamiento es una **limitación inherente de los métodos basados en árboles** para series con tendencia de largo plazo.

### Consistencia XAI inter-modelo (Spearman ρ)

| Par de modelos | ρ | Interpretación |
|---|---|---|
| LSTM vs Transformer | **0.748** | Concordancia moderada-alta |
| LSTM vs Random Forest | 0.255 | Concordancia baja |
| Transformer vs Random Forest | 0.382 | Concordancia moderada-baja |

LSTM y Transformer comparten una lógica temporal similar (ambos son modelos de secuencia), mientras que el RF opera con una lógica radicalmente distinta (modelo tabular sin memoria explícita).

### Significancia estadística

Los tres modelos producen matrices de importancia estadísticamente distinguibles de explicaciones aleatorias (test Mann-Whitney, p ≈ 0, todas las comparaciones superan el umbral de 0.05 por un margen de decenas de órdenes de magnitud).

### Patrones de explicabilidad identificados

**LSTM — memoria distribuida:**  
Distribuye importancia de forma gradual a lo largo de toda la ventana de 20 días, con una decaída suave desde t-19 hacia t-0. Las variables más relevantes son los precios OHLC y las medias móviles SMA-5, SMA-10 y SMA-20. El mecanismo de atención aprovecha el contexto histórico completo para su predicción.

**Transformer — atención en la semana reciente:**  
Con attention pooling, concentra la importancia principalmente en los últimos 7 días (t-0 a t-6), con énfasis en High, Low y SMA-10. Muestra mayor especialización temporal que el LSTM pero sin la degradación artificial que producía el pooling por último token.

**Random Forest — estrategia de último precio:**  
Importancia casi exclusivamente en Close y Low en t-0 y t-1. Essencialmente implementa la heurística "el precio de mañana ≈ el precio de hoy", sin memoria temporal real. Esta estrategia funciona bien en períodos de baja volatilidad pero colapsa cuando los precios salen del rango de entrenamiento.

**Hallazgo transversal:**  
MACD, MACD Signal, RSI-14, BB_width, Log_Return y Volume_ratio presentan importancia cercana a cero en los tres modelos. Esto sugiere que estos indicadores derivados son redundantes dado que los modelos ya tienen acceso a los precios base y las SMAs desde las cuales se calculan.

---

## Decisiones de diseño y correcciones metodológicas

Durante el desarrollo se identificaron y corrigieron los siguientes problemas:

| Problema | Impacto | Solución aplicada |
|---|---|---|
| Leakage en validación: split de ventanas post-hoc | Early stopping con val set que comparte historia con train | Split temporal de DataFrames antes de construir ventanas (`prepare_data_with_val`) |
| Transformer con last-token pooling (`x[:, -1]`) | Explicaciones artificialmente concentradas en t-0 por diseño arquitectónico, no por aprendizaje | Reemplazado por attention pooling aprendido |
| Scaler ajustado solo en train (70%) | 99% de muestras de test fuera del rango [0,1] por tendencia de precios | Scaler ajustado sobre período de desarrollo completo (80%) |
| Alphas con V=5 (solo OHLCV) | Gráficos incompletos y resultados stale | Regeneración con las 15 features correctas |
| Sin validación del vector de referencia | Errores silenciosos si reference contiene NaN | Asserts en el constructor de `TemporalPerturbationEngine` |
| `explain.py` sin verificar checkpoints | `FileNotFoundError` sin mensaje útil | Verificación explícita antes de `torch.load` |

---

## Trabajo futuro

<<<<<<< HEAD
- Python 3.10+
- PyTorch ≥ 2.0
- scikit-learn ≥ 1.3
- Ver `requirements.txt` para la lista completa
- 
=======
### Preprocesamiento y datos

- **Normalización por ventana (within-window normalization):** normalizar cada ventana de T días relativa a su primer elemento elimina la tendencia de largo plazo de forma local, haciendo el problema estacionario sin necesidad de ajustar un scaler global. Es la solución más robusta para series financieras con tendencia fuerte.
- **Target de retorno logarítmico:** cambiar el objetivo de predicción de `Close` a `Log_Return` haría la variable objetivo estacionaria por definición. Sin embargo, los retornos diarios tienen muy baja autocorrelación (eficiencia de mercado), lo que dificulta el aprendizaje.
- **Ampliación del universo de activos:** aplicar el framework a otros índices (NASDAQ, FTSE, Nikkei) o activos (Bitcoin, oro) para evaluar generalización.

### Modelos

- **Modelos con memoria más larga:** probar con WINDOW_SIZE=40 o 60 días para capturar ciclos mensuales.
- **BiLSTM:** una LSTM bidireccional, usada en contexto offline (explicabilidad post-hoc), podría revelar patrones temporales más ricos.
- **Modelos más recientes:** comparar con Temporal Fusion Transformer (TFT) o TimesNet, que están diseñados específicamente para series temporales.
- **Ensemble de modelos:** combinar predicciones de LSTM y Transformer puede reducir varianza.

### Framework XAI

- **Análisis de sensibilidad al baseline de referencia:** comparar las matrices α obtenidas usando `reference = media`, `reference = mediana`, `reference = 0` y `reference = valor más frecuente`. La elección del baseline tiene impacto en las importancias finales y no está estudiada sistemáticamente en la literatura de XAI temporal.
- **Comparación con métodos establecidos:** contrastar las matrices α del framework propuesto con SHAP (ShapleyValueSampling) y LIME temporal. Evaluar si el motor de perturbaciones converge a soluciones similares con menor costo computacional.
- **Análisis de la relación entre pesos de atención y α:** para LSTM y Transformer, los pesos del mecanismo de atención interno son una forma de explicabilidad nativa. Comparar sistemáticamente si esos pesos correlacionan con las α del motor de perturbaciones.
- **Selección estratificada de muestras para explicar:** en lugar de muestras aleatorias del test set, seleccionar 25 muestras de cada cuartil de error de predicción. Esto permitiría estudiar si el framework genera explicaciones diferentes para predicciones fáciles vs. difíciles.
- **Batch processing de perturbaciones con GPU:** la generación de T×V+1 perturbaciones por muestra puede paralelizarse en GPU con sub-batches, reduciendo el tiempo de `explain.py` de minutos a segundos.

### Evaluación

- **Métricas financieras relevantes:** el MSE en espacio normalizado no es directamente interpretable. Agregar métricas como *directional accuracy* (¿el modelo predice correctamente la dirección del movimiento?), MAPE en puntos del índice y Sharpe ratio simulado.
- **Corrección de múltiples comparaciones:** con tres tests de significancia paralelos, aplicar corrección de Bonferroni para controlar el FWER (Family-Wise Error Rate).
- **Análisis de estabilidad temporal (rolling window):** entrenar sobre 2010-2018, evaluar en 2019; luego entrenar sobre 2010-2019, evaluar en 2020; etc. Medir si los patrones de importancia α son estables a lo largo del tiempo o varían con el régimen de mercado.

### Análisis de resultados

- **Análisis por régimen de mercado:** segmentar el test set en períodos alcistas (*bull market*), bajistas (*bear market*) y de alta volatilidad (p.ej., COVID marzo 2020, caída 2022). Estudiar si las matrices α cambian según el régimen, lo que podría revelar que distintos factores son relevantes en distintos contextos de mercado.
- **Análisis de redundancia de features:** los indicadores MACD, RSI, BB_width, Log_Return y Volume_ratio muestran importancia casi nula en los tres modelos. Un análisis de correlación e información mutua entre los 15 features podría confirmar si son realmente redundantes dado Close y las SMAs, y justificar una reducción del espacio de features.
- **Cuantificación del costo computacional vs. calidad de explicaciones:** medir el trade-off entre el número de perturbaciones (T×V) y la estabilidad de las matrices α, para determinar si es posible usar un subconjunto de perturbaciones con pérdida mínima de precisión.

---

## Referencia rápida de hiperparámetros

| Parámetro | Valor | Descripción |
|---|---|---|
| WINDOW_SIZE | 20 | Días de historia por ventana |
| HORIZON | 1 | Pasos hacia adelante a predecir |
| TRAIN_RATIO | 0.80 | Fracción de desarrollo (train + val) |
| VAL_RATIO | 0.10 | Fracción de val dentro del desarrollo |
| EPOCHS | 100 | Máximo de épocas (con early stopping) |
| PATIENCE | 15 | Épocas sin mejora antes de parar |
| LR | 1e-3 | Learning rate inicial (AdamW) |
| LSTM_HIDDEN | 128 | Dimensión oculta LSTM |
| TF_D_MODEL | 64 | Dimensión del Transformer |
| TF_NHEAD | 4 | Cabezas de atención |
| RF_N_ESTIMATORS | 200 | Árboles en el Random Forest |
| N_EXPLAIN | 100 | Muestras del test a explicar |
>>>>>>> b104700 (Correcciones metodológicas, arquitectónicas y documentación completa)
