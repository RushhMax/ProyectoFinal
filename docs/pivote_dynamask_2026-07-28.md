# Pivote hacia DynaMask — sesión 2026-07-28 (continuar mañana)

## Contexto / por qué

La profesora dijo que el paper debería acortarse/delimitarse tipo DynaMask:
enfocar en menos modelos y encontrar una limitación específica de DynaMask
para mejorar desde ahí, en vez de comparar contra ~15 trabajos relacionados.

Pregunta clave de la profesora (parafraseada): **"¿qué hiciste para que el
framework fuera bueno en RF? ¿tuviste alguna consideración extra?"** — su
punto es que el framework debe tener consideración específica por modelo,
no ser genérico aplicado igual a los 3.

## Hallazgo en el código (confirma el punto de la profesora)

`framework/explainer.py:31-59` y `framework/perturbation.py`: la única
diferencia entre modelos es `model_type` ('pytorch' vs 'sklearn') para
saber cómo llamar `.predict()`. El mecanismo de perturbación (sustituir por
vector de referencia) es **idéntico** para LSTM, Transformer y RF. Cero
consideración específica por arquitectura hoy.

`models/random_forest.py`: wrapper genérico, aplana `(N,T,V)` a `(N,T*V)`
antes de llamar a sklearn. No expone nada especial, pero
`RandomForestRegressor` sí tiene `.estimators_` (los árboles individuales)
disponible sin modificar el modelo.

## Dos hallazgos propios ya existentes que apuntan a que RF necesitaba trato distinto

1. RF no puede extrapolar más allá del rango de target visto en
   entrenamiento (por eso se cambió a log-return; ver memoria
   `project_pfc`/`project_paper_related_work`).
2. RF es mucho más sensible al vector de referencia (ρ=0.81/0.67) que
   LSTM/Transformer (ρ≈0.99) — resultados ya en "Robustez Adicional"
   (`results/additional_results.json`).

## Idea nueva para RF (la más prometedora, sin explorar aún)

**Descomposición por árbol**: usar `rf.model.estimators_` para calcular el
delta de oclusión árbol por árbol y agregar media ± desviación estándar,
dando un intervalo de confianza natural para cada atribución — algo que
LSTM/Transformer no pueden ofrecer sin trabajo extra (ensemble de seeds,
MC dropout, etc.). Es una consideración anclada en la estructura del
modelo (bagging), no solo un hiperparámetro ajustado.

Otras ideas más débiles/secundarias:
- Reference vector adaptado a RF (percentiles del training set en vez de
  media global, dado que RF corta por umbrales y es sensible a caer fuera
  de la distribución vista).
- Usar `feature_importances_` nativo de RF (MDI/permutation) como chequeo
  de validez cruzada — único entre los 3 modelos porque RF tiene una señal
  de importancia nativa; LSTM/Transformer no.

## Limitaciones de DynaMask verificadas contra el PDF original (más allá de "no gradientes")

Fuente: PDF completo obtenido vía ar5iv
(https://ar5iv.labs.arxiv.org/html/2106.05303), Crabbé y van der Schaar,
ICML 2021. Verificado contra el texto real, no solo resúmenes.

1. **Nunca se valida en datos financieros.** La introducción menciona
   finanzas como motivación, pero los 3 experimentos reales son: sintético
   caja blanca (ARMA), sintético caja negra (HMM), y MIMIC-III (mortalidad
   clínica). Cero finanzas. Limitación válida porque conecta con algo que
   el propio paper reclama (motivación financiera) — mismo criterio de
   validez usado en Trabajos Relacionados (ver memoria
   `project_paper_related_work`).

2. **Optimización costosa e hiperparametrizada sin análisis de sensibilidad
   propio.** La máscara se ajusta por instancia con ~1000 épocas de
   descenso de gradiente, 2 forward passes por época. 5 hiperparámetros:
   restricción de área $a$, $\lambda_a$ (regulador de tamaño, init ~0.1,
   dilatado por $\delta$≈1000), $\lambda_c$ (continuidad temporal, penaliza
   $\sum|m_{t+1,i}-m_{t,i}|$), learning rate (típico 1), momentum (típico
   1). El propio paper dice explícitamente "no discute failure modes" ni
   sensibilidad a estos valores. FO no tiene ningún hiperparámetro y es una
   sola pasada batch determinista.

3. **Los operadores de perturbación asumen suavidad/estacionariedad
   local.** Tres operadores: blur gaussiano ($\pi^g$, asume suavidad
   temporal local, complejidad $O(d_X T^2)$), fade-to-moving-average
   ($\pi^m$, asume estacionariedad en ventana $2W+1$, $O(d_X T W)$),
   fade-to-past ($\pi^p$, para forecasting, solo usa pasado). Todos asumen
   ventanas localmente suaves/estacionarias — choca con lo que ya
   documentó la tesis en "Robustez Adicional" (consistencia cae en régimen
   bajista 2022, mercados financieros tienen saltos de volatilidad que
   violan esa suposición). Puente natural a resultados ya existentes.

## Búsquedas hechas (para no repetir)

- WinTSR (Islam y Fox, arXiv 2412.04532, **preprint sin arbitrar**, marzo
  2025): lista 4 limitaciones de métodos tipo DynaMask (enfoque en
  clasificación no regresión, baselines desactualizados, datasets
  sintéticos simples, requieren entrenar/optimizar otro modelo por
  instancia). Útil como apoyo pero no como evidencia central (no
  arbitrado, mismo problema que GS-SHAP/TsSHAP ya documentado).
- "Interpretability in Deep Time Series Models Demands Semantic Alignment"
  (arXiv 2602.02239, ICML 2026, arbitrado): crítica conceptual de que
  métodos de masking reducen todo a un escalar sin alineación semántica.
  No menciona DynaMask directamente, ángulo distinto (semántico, no
  mecanístico), probablemente no aplica directo a este pivote.
- Yadav y Subbian (arXiv 2506.19035) — ya conocido y citado en el paper,
  no es nuevo.

## Tensión detectada (importante, no resuelta del todo)

El paper YA está bastante centrado en DynaMask (`sec:avance2`/tabla
`tab:advances`, sección "Brecha con DynaMask" línea ~1263 de `paper.tex`,
replica el experimento exacto de Crabbé con las mismas métricas
AUP/AUR/$I_M$/$S_M$). RF es la pieza que sostiene el avance principal
actual (DynaMask no corre en RF, el framework sí). La usuaria decidió NO
sacrificar RF del alcance — por eso el plan es delimitar A RF
específicamente (no a LSTM+Transformer como se pensó al inicio de la
sesión), y sumar la descomposición por árbol como la "consideración extra"
que pide la profesora, más los 2 puntos nuevos de DynaMask (validación
financiera, costo/hiperparámetros) como refuerzo.

## Próximo paso (sin decidir aún cuál ir primero)

Dos opciones que quedaron abiertas al cortar la sesión:
1. Explorar si la descomposición por árbol es viable con
   `saved_models/rf_logret.pkl` ya entrenado (implementar, correr, ver si
   reduce la sensibilidad ρ=0.81 hacia algo más cercano a 0.99).
2. Medir el costo computacional real de DynaMask vs FO (usando el
   `README.md`/código de https://github.com/JonathanCrabbe/Dynamask) para
   tener el número concreto del punto 2 de limitaciones.

**No se ha escrito ni modificado nada en `paper.tex` todavía** — toda esta
sesión fue exploración/investigación, ningún cambio de contenido aplicado.
