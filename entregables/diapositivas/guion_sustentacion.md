# Guion de sustentación (~30 minutos)

## 1. Portada

Buenos días. Mi nombre es Rushell Vanessa Zavalaga Orozco y voy a presentar mi trabajo de tesis, "Framework de Explicabilidad Agnóstica al Modelo para la Predicción de Series Temporales Financieras".

## 2. Motivación (1/2)

Los modelos de aprendizaje profundo predicen con alta precisión series temporales financieras. Pero no revelan qué información del historial determinó cada predicción. Este no es un problema cosmético, tiene consecuencias regulatorias concretas. El SR 11-7 de la Reserva Federal estadounidense obliga a los bancos a demostrar que entienden las capacidades y los límites de un modelo antes de usarlo. El Reglamento de Inteligencia Artificial de la Unión Europea va más allá para ciertos usos financieros, como la evaluación crediticia, y exige que el modelo sea lo bastante transparente para que quien lo usa pueda interpretar sus resultados. Sin explicabilidad no es posible detectar sesgos ni fallos del modelo. Un modelo que no puede explicarse no puede auditarse, la opacidad impide distinguir si aprendió una relación económica real o una coincidencia espuria en los datos de entrenamiento.

## 3. Motivación (2/2)

Un modelo entrenado sobre datos financieros históricos hereda los sesgos de ese historial. Por eso, a lo largo de este trabajo, reporto también los resultados que no favorecen a mi propuesta, como que ningún modelo entrenado supera una predicción ingenua de persistencia, o que una consideración específica que investigué para Random Forest no confirmó mi hipótesis inicial. Prefiero reportar esos resultados en vez de omitirlos. Además, al no requerir gradientes, mi framework reduce la barrera de cómputo para auditar modelos, un punto de acceso equitativo para instituciones o reguladores que no cuentan con la infraestructura que exigen métodos como Shapley Value Sampling o las máscaras aprendidas por gradiente. La pregunta central que motiva este trabajo es la siguiente: ¿qué variable fue relevante para una predicción, y cuándo, en el historial, lo fue?

## 4. El Problema

Formulo el problema que resuelve esta tesis como un problema de cómputo con dos requisitos que ningún método previo satisface a la vez. Primero, que el mecanismo de explicación funcione sobre arquitecturas de naturaleza computacional distinta, sin acceder a gradientes ni a la arquitectura interna del modelo. Segundo, que ese mecanismo tenga un costo fijo por instancia, evaluable en una sola pasada, sin resolver una optimización propia cada vez que se explica una predicción.

DynaMask, de Crabbé y Van Der Schaar, es el antecedente más directo. Logra coherencia temporal optimizando una máscara de perturbación por descenso de gradiente, hasta mil épocas y cinco hiperparámetros en su formulación original. Eso excluye por diseño a los modelos no diferenciables, como Random Forest. Lo relevante es que esto no se resolvió después. Sus dos extensiones directas, ExtrMask, publicada en ICML 2023, y ContraLSP, publicada en ICLR 2024, corrigen la calidad de la perturbación, pero ninguna de las dos relaja la dependencia del gradiente, y ninguna de las tres se ha validado sobre series financieras pese a mencionarlas como motivación en su introducción. Cinco años después de DynaMask, ningún método de esta línea de trabajo funciona sin gradientes ni se ha validado en finanzas.

## 5. Trabajos Relacionados: Dos Ejes

Organizo los trabajos relacionados en dos ejes. El primer eje agrupa métodos de perturbación agnósticos que incorporan de algún modo la dimensión temporal, pero sin agnosticismo total. Huang et al., con ShapeX, aplican valores de Shapley sobre shapelets en vez de puntos individuales, pero está diseñado únicamente para clasificación, no para regresión continua. Yadav y Subbian diagnostican por qué los métodos de gradiente, oclusión y permutación fallan específicamente en escenarios de predicción dinámica.

El segundo eje agrupa frameworks explícitamente temporalmente conscientes y agnósticos. Tonekaboni et al., con FIT, y Leung et al., con WinIT, están pensados para predicción en línea sobre datos clínicos, no para una ventana única como la que yo trabajo. Crabbé y Van Der Schaar, con DynaMask, producen el mismo formato de salida T por V que yo, pero requieren gradientes. Bento et al., con TimeSHAP, están limitados a modelos secuenciales. Raykar et al., con TsSHAP, trabajan sobre características predefinidas, sin resolución por celda individual. Roshinta et al. y Kim y Park, con GS-SHAP, son agnósticos, pero agregan variables mediante componentes principales o grupos, y evalúan un único tipo de modelo.

## 6. La Brecha

Ningún trabajo previo satisface simultáneamente tres condiciones. Primero, agnosticismo real al modelo, sin gradientes y sin estructura secuencial obligatoria. Segundo, resolución completa por celda, es decir por variable y por paso de tiempo, sin agrupar variables ni segmentos. Tercero, verificación empírica de que las explicaciones cambian entre modelos de distinta naturaleza computacional, siguiendo el criterio que establecieron Adebayo et al. en 2018. Mi hipótesis de trabajo es que es posible construir un método de perturbación temporalmente ordenado y agnóstico al modelo que produzca explicaciones consistentes con el comportamiento de cualquier arquitectura, y cuya salida central es una matriz alpha, en el espacio de los números reales, de dimensión T por V.

## 7. Pregunta de Investigación e Hipótesis

Mi pregunta de investigación es la siguiente: ¿es posible atribuir la importancia de cada variable y cada paso de tiempo dentro de una ventana de entrada, de forma agnóstica al modelo, capturando el comportamiento real de arquitecturas de distinta naturaleza computacional? Mi hipótesis es que sí es posible construir un método de perturbación temporalmente ordenado, agnóstico al modelo, que produzca explicaciones consistentes con el comportamiento de cualquier arquitectura.

## 8. Objetivo General

Mi objetivo general es proponer un framework de explicabilidad agnóstico al modelo que atribuya importancia por variable y paso de tiempo en la predicción de series temporales financieras, sin depender de la estructura interna del modelo explicado.

## 9. Objetivos Específicos

Para lograrlo, planteo cuatro objetivos específicos. Diseñar una interfaz uniforme de predicción, mediante un patrón adaptador de dos niveles, que desacople el motor de perturbaciones de la estructura interna de cualquier modelo. Formular un motor de perturbaciones temporales que atribuya importancia a nivel de celda, preservando el orden y la coherencia temporal del resto de la ventana. Entrenar modelos de distinta naturaleza computacional, LSTM, Transformer y Random Forest, sobre el S&P 500 como banco de pruebas del framework. Y evaluar el framework frente al estado del arte agnóstico, en precisión de identificación, consistencia entre modelos y velocidad de cómputo.

## 10. Contribución: Interfaz Uniforme de Predicción

Mi contribución de diseño es un patrón adaptador de dos niveles. El motor de perturbaciones temporales, que en mi implementación se llama TemporalPerturbationEngine, nunca accede al modelo directamente, solo conoce una función f que va de un tensor de N por T por V a un vector de N predicciones. En el primer nivel, cada modelo implementa su propio wrapper: el de Random Forest aplana la ventana de T por V a T por V en una sola dimensión antes de invocar al regresor de scikit-learn; los de LSTM y Transformer convierten el arreglo a un tensor de PyTorch y ejecutan la inferencia sin gradientes. En el segundo nivel, el explicador construye esa función f y la inyecta en el motor, de modo que cualquier modelo externo puede conectarse sin modificar ningún componente existente del framework. Esto implementa una inversión de dependencias: el framework no depende de los modelos, son los modelos los que se adaptan al framework.

## 11. Datos y Modelos

Trabajo con el índice S&P 500, en el periodo 2010 a 2024, tres mil setecientos cincuenta y cuatro días hábiles. Uso quince variables por paso de tiempo, cinco variables base de precio y volumen, apertura, máximo, mínimo, cierre y volumen, y diez indicadores técnicos derivados, medias móviles, MACD, RSI, ancho de bandas de Bollinger, rango verdadero promedio, retorno logarítmico y volumen relativo. Las ventanas de entrada tienen veinte pasos de tiempo, con horizonte de predicción de un día. El split respeta el orden temporal, ochenta por ciento para desarrollo y veinte por ciento para prueba, sin mezclar información del futuro hacia el pasado. Entreno tres modelos heterogéneos bajo la misma interfaz: un LSTM de dos capas con ciento veintiocho unidades ocultas y veinte por ciento de dropout; un Transformer de dos capas con cuatro cabezas de atención y agregación aprendida mediante attention pooling; y un Random Forest de doscientos árboles de decisión.

## 12. Motor de Perturbaciones Temporales (1/2)

El componente central de mi propuesta es el motor de perturbaciones temporales. Para cada posición, definida por un paso de tiempo t y una variable v, dentro de la ventana de entrada, reemplazo el valor original por una referencia r sub v, que es la media de esa variable sobre el conjunto de entrenamiento, y mido el cambio absoluto en la predicción del modelo. Esto se conoce como oclusión. Elegí oclusión, en vez de una máscara aprendida por gradiente, por dos razones. No requiere gradientes, así que funciona igual sobre Random Forest que sobre los modelos neuronales. Y perturba una sola celda a la vez, dejando el resto de la ventana exactamente en su valor y posición temporal observados, lo cual preserva la coherencia causal de la serie.

## 13. Motor de Perturbaciones Temporales (2/2)

Calcular T por V inferencias individuales sería costoso e innecesario. En su lugar, construyo un único batch de T por V más uno ventanas, y evalúo ese batch en una sola llamada al modelo. El costo escala como T por V, no crece de forma independiente por cada dimensión. Finalmente, normalizo la matriz resultante con norma L1, de modo que cada celda de la matriz alpha estimada indica la proporción del cambio total en la predicción que es atribuible a esa variable, en ese paso de tiempo específico.

## 14. Pipeline del Framework

Este diagrama resume el flujo completo del framework en seis etapas: los datos de entrada, la preparación y construcción de ventanas, el modelo predictivo, la interfaz uniforme de predicción, el motor de perturbaciones temporales, y la matriz de importancia resultante. Las etapas cuatro y cinco, la interfaz y el motor, constituyen el núcleo agnóstico de mi propuesta.

## 15. Validación con Ground Truth Sintético

Antes de aplicar el framework sobre datos reales, donde no conozco la verdad de fondo, lo valido en un escenario donde sí la conozco. Construyo un regresor de caja blanca cuya predicción depende únicamente de un subconjunto conocido de celdas salientes, con ventanas de veinte pasos de tiempo y quince variables, diez repeticiones, y dos escenarios, uno donde pocas variables son relevantes y otro donde pocos pasos de tiempo lo son. Bajo este banco de pruebas, mi método, al que llamo FO, alcanza un área bajo la curva de precisión de 0,9960, igualando a Shapley Value Sampling, Integrated Gradients y Feature Permutation, y superando el 0,97 que reporta DynaMask en su propio benchmark, sin requerir gradientes y ejecutándose también sobre Random Forest.

## 16. Verificación de Agnosticidad sobre S&P 500

Sobre cien instancias reales del conjunto de prueba, mido la correlación de Spearman entre las matrices de importancia que produce el framework para cada par de modelos. Entre LSTM y Transformer, la correlación es de 0,748. Entre LSTM y Random Forest, cae a 0,255. Entre Transformer y Random Forest, es de 0,382. Esta diferencia no es ruido, las tres correlaciones son estadísticamente significativas frente a un baseline aleatorio, con una prueba de Mann-Whitney U que da un valor p menor a diez elevado a la potencia menos trescientos, en los tres modelos. El patrón que observo, mayor consistencia entre modelos neuronales que entre un modelo neuronal y el modelo de ensamble, es exactamente lo que esperaría si las explicaciones capturan el comportamiento específico de cada arquitectura, y no son un artefacto del procedimiento de explicación.

## 17. Resultados: Importancia por Variable

LSTM y Transformer priorizan las medias móviles de cinco, diez y veinte días, y el precio máximo del día. Random Forest concentra su importancia casi por completo en el precio de cierre y el precio mínimo. Los indicadores de oscilación, MACD, RSI y el ancho de bandas de Bollinger, quedan cerca de cero en los tres modelos. Estos resultados hay que interpretarlos con cautela, y los retomo en la diapositiva de retorno logarítmico, porque el target de precio de cierre tiene una autocorrelación fuerte con el precio del día anterior.

## 18. Resultados: Patrón Temporal de Importancia

En el eje temporal, LSTM muestra un patrón decreciente del pasado hacia el presente, con una razón entre el máximo y el mínimo de quince punto siete, coherente con una memoria distribuida a lo largo de la ventana. Transformer muestra un perfil casi uniforme, con una razón de uno punto cinco, coherente con su mecanismo de attention pooling, que aprende a ponderar toda la secuencia. Random Forest concentra casi toda su importancia en el último paso de tiempo, con una razón de ochocientos sesenta, porque no modela la secuencia, cada celda aplanada es, para este modelo, una variable tabular más entre trescientas.

## 19. Resultados: Matrices de Importancia Comparadas

Este mapa de calor resume visualmente los dos resultados anteriores. En LSTM y Transformer, las filas correspondientes a las medias móviles concentran los valores más altos, distribuidos a lo largo del tiempo. En Random Forest, la importancia se concentra en la columna correspondiente al paso de tiempo más reciente, sobre las variables de precio de cierre y mínimo.

## 20. Robustez: Target de Retorno Logarítmico

El target de precio de cierre tiene una autocorrelación trivial con el precio actual. Para comprobar que mis resultados no dependen de esa autocorrelación, repito todo el pipeline usando como target el retorno logarítmico del día siguiente, una variable que rompe esa autocorrelación por construcción. El resultado es contundente. Ningún modelo entrenado, ni LSTM, ni Transformer, ni Random Forest, supera un baseline de persistencia ingenua que predice un retorno igual a cero. Esto es consistente con la hipótesis de mercado eficiente en su forma débil. Este resultado no es una falla de mi framework de explicabilidad, cuyo objetivo es explicar el comportamiento del modelo entrenado, sea este bueno o malo prediciendo, es una confirmación honesta de que la tabla de métricas original medía en gran parte la capacidad de los modelos de reproducir el nivel de precio actual, no de anticipar su variación futura. Aun así, el patrón de consistencia entre modelos se reproduce bajo este target: LSTM y Transformer correlacionan en 0,541, frente a 0,187 entre LSTM y Random Forest, y 0,311 entre Transformer y Random Forest, con significancia estadística en los tres casos.

## 21. Robustez Adicional

Someto al framework a tres pruebas adicionales, sin reentrenar ningún modelo. La exactitud direccional, es decir si el modelo acierta si el precio sube o baja, se mantiene entre cuarenta y seis y cincuenta por ciento en los tres modelos y ambos targets, indistinguible del azar. La sensibilidad al vector de referencia, comparando media, mediana y cero, muestra que LSTM y Transformer son altamente estables, con correlaciones sobre 0,88, mientras que Random Forest es notablemente más sensible, con correlaciones que bajan hasta 0,666. Y por régimen de mercado, comparando el periodo bajista de 2022 con el periodo alcista de 2023 y 2024, las correlaciones son algo menores durante el régimen bajista, pero el orden relativo entre modelos se preserva en ambos regímenes.

## 22. Consideración Específica para Random Forest

La prueba anterior muestra que Random Forest es más sensible que los modelos neuronales al vector de referencia. Esto me llevó a investigar si existe una consideración anclada en la estructura propia de Random Forest, no solo un valor externo distinto, que reduzca esa sensibilidad. Hooker, Mentch y Zhou muestran que los métodos de tipo permutar y predecir fuerzan al modelo a extrapolar cuando la perturbación rompe la dependencia entre variables, precisamente porque el valor de reemplazo cae fuera de la región de datos que el modelo observó. Random Forest ofrece una forma directa de acotar esa región, porque cada árbol define hojas, particiones del espacio de entrada donde el modelo trata a las observaciones como equivalentes. Implementé y evalué una referencia condicionada a la hoja, calculada por instancia, promediando los ejemplos de entrenamiento que caen en la misma hoja que la ventana a explicar, ponderados por cercanía mediante un núcleo gaussiano.

El resultado fue negativo. Sobre el banco de pruebas sintético, el área bajo la curva de precisión y de recall no cambiaron de forma significativa frente a la referencia media global. Sobre el S&P 500 real, el efecto fue negativo, no neutro: la consistencia con LSTM y Transformer, que con la referencia media es de 0,184 y 0,311, cayó a valores negativos, menos 0,111 y menos 0,232, con la referencia condicionada a la hoja. La explicación que encontré es estructural. La referencia media que uso para los tres modelos es externa a cada uno, proviene de los datos. La referencia condicionada a la hoja es interna, se construye a partir de la propia estructura de árboles del Random Forest que estoy explicando, y eso rompe el criterio común entre modelos del que depende la comparación de agnosticismo. Por esta razón, mantengo una referencia externa e idéntica en su naturaleza para los tres modelos; esta alternativa que investigué empeora la comparabilidad en lugar de mejorarla, y ese resultado negativo respalda la decisión de diseño que ya tenía.

## 23. Ablación: ¿Importa Respetar el Orden Temporal?

Para verificar que respetar el orden temporal realmente importa, y no es solo una preferencia de diseño, diseñé una variante de control negativo. En vez de reemplazar la celda por la referencia fija, la sustituyo por el valor de la misma variable en otro instante, elegido al azar dentro de la misma ventana, violando el orden temporal a propósito. En el benchmark sintético, que es determinista, esta variante no se distingue de mi método, porque la función de caja blanca no tiene ninguna noción de coherencia temporal. Pero sobre datos reales, el daño es evidente. La consistencia entre LSTM y Transformer cae de 0,748 a 0,487. La consistencia entre LSTM y Random Forest cae de 0,255 a menos 0,055, prácticamente nula. La fidelidad se reduce hasta treinta y ocho veces en Random Forest. Y las explicaciones dejan de ser reproducibles entre corridas distintas, con una correlación de apenas 0,55.

## 24. Comparación en Condiciones Idénticas: FO vs. SVS

Comparo mi método contra Shapley Value Sampling, el único otro método agnóstico al modelo que produce una matriz T por V ejecutable sobre los tres modelos, bajo condiciones experimentales idénticas: mismos datos, mismos modelos, mismo hardware. Ocluyo las celdas que cada método identifica como más importantes y mido cuánto se desplaza la predicción resultante. Mi método resulta más fiel en los modelos neuronales, con un desplazamiento de 0,5033 frente a 0,4891 en LSTM, y 0,4518 frente a 0,4352 en Transformer. Shapley Value Sampling es mejor específicamente en Random Forest, 0,3212 frente a 0,3047, porque su muestreo de coaliciones captura mejor las interacciones combinatorias entre árboles, algo que la oclusión celda por celda no está diseñada para capturar.

## 25. Velocidad: FO vs. Estado del Arte

En velocidad, mi método ejecuta cada explicación en 17,6 milisegundos sobre LSTM, 13,8 sobre Transformer, y 112 sobre Random Forest. Frente a Shapley Value Sampling, bajo las mismas condiciones, esto representa una aceleración de entre veintidós y treinta veces, porque mi método requiere exactamente trescientas una pasadas hacia adelante, mientras que Shapley Value Sampling necesita miles de evaluaciones por muestreo de permutaciones. Frente a TimeSHAP, la aceleración es de ciento cincuenta y nueve punto ocho veces, y frente a GS-SHAP, de sesenta y siete punto ocho veces. Estas dos últimas cifras provienen de una fuente externa, un preprint todavía sin arbitrar de Kim y Park, así que las presento como comparación de contexto, no como evidencia central de mi tesis.

## 26. Resumen de Avances frente al Estado del Arte

En síntesis, frente a DynaMask obtengo el mismo o mejor AUP, cero coma nueve nueve seis cero frente a cero coma noventa y siete, y corro también sobre Random Forest, algo que DynaMask no puede hacer. Frente a TimeSHAP y GS-SHAP, obtengo resolución completa por celda en vez de una salida agrupada, y una velocidad uno o dos órdenes de magnitud mayor. En conjunto, mi propuesta logra el mismo o mejor desempeño de identificación, resolución completa por celda, funciona sobre modelos no diferenciables, y es órdenes de magnitud más rápida que las alternativas basadas en valores de Shapley.

## 27. Conclusiones

Este trabajo presenta un framework de explicabilidad temporal agnóstico al modelo para la predicción de series temporales financieras. Su contribución central es un patrón adaptador de dos niveles que desacopla el motor de perturbaciones de cualquier backend de modelo específico, permitiendo que cualquier modelo con la firma unificada se conecte sin modificar el motor. Sobre datos sintéticos, mi método iguala a Shapley Value Sampling en precisión de identificación. Sobre el S&P 500, supera a Shapley Value Sampling en fidelidad sobre los modelos neuronales, siendo entre veintidós y treinta veces más rápido, mientras que Shapley Value Sampling resulta mejor específicamente en Random Forest. El framework produce atribuciones estadísticamente no triviales y consistentes con la familia arquitectónica de cada modelo, con una correlación de 0,748 entre modelos neuronales frente a menos de 0,40 con el modelo de ensamble, un resultado robusto frente al target evaluado, al vector de referencia, y al régimen de mercado.

## 28. Limitaciones

Este trabajo tiene cuatro limitaciones que reconozco explícitamente. La variable objetivo de precio de cierre tiene alta autocorrelación con el precio actual, lo cual mitigo evaluando también el retorno logarítmico. El escalador MinMax, ajustado sobre el periodo 2010 a 2021, no cubre los máximos históricos del periodo 2022 a 2024, lo que perjudica en particular a Random Forest. El benchmark sintético usa una función determinista sin ruido, por lo que no discrimina bien entre métodos; con ruido aditivo, mi método y Shapley Value Sampling sí se diferencian, siendo Shapley Value Sampling más robusto por promediar sobre permutaciones. Y el alcance de mi evaluación se limita a un único índice financiero, el S&P 500, aunque sí lo evalúo frente a distintas referencias, regímenes de mercado y un target alternativo.

## 29. Trabajo Futuro

Como trabajo futuro planteo cuatro líneas. Incorporar perturbación bidireccional, por encima y por debajo de la referencia, para recuperar el signo de la atribución y no solo su magnitud. Replicar el mismo pipeline sobre otro índice bursátil, como el NASDAQ Composite, cambiando un único parámetro de configuración. Correlacionar la volatilidad realizada por ventana con la concordancia inter-modelo de la matriz alpha, instancia por instancia, y no solo por régimen agregado. Y extender el motor a perturbación por bloques de celdas, para escalar a ventanas o conjuntos de variables mucho mayores.

## 30. Declaración de Uso de Inteligencia Artificial

Durante la elaboración de este trabajo utilicé asistentes de inteligencia artificial generativa como apoyo en tareas puntuales de edición de redacción y verificación de referencias bibliográficas, y para la generación de código auxiliar de experimentos adicionales de robustez, siempre bajo mi supervisión directa. Todas las cifras que reporté en esta sustentación provienen de ejecución real del código sobre los datos descritos. El diseño del framework, la formulación matemática del motor de perturbaciones, la interpretación de los resultados y las conclusiones son responsabilidad exclusiva de mi persona.

## 31. Referencias Principales

Estas son mis referencias principales. Quedo atenta a sus preguntas.

---

# Respuestas a preguntas previsibles del jurado

**¿Por qué no usar simplemente SHAP o LIME?**

SHAP y LIME fueron formulados asumiendo variables independientes. Al aplicarse sobre series temporales, tratan cada paso de tiempo como una variable aislada, ignorando que las observaciones forman una secuencia con dependencias estructurales, tal como lo documentan Tonekaboni et al. y Huang et al.

**¿Por qué su framework es una contribución y no una reimplementación de DynaMask?**

Porque DynaMask, y sus dos extensiones directas publicadas en 2023 y 2024, requieren gradientes por diseño para optimizar su máscara de perturbación. Mi motor no requiere gradientes, y eso es precisamente lo que permite ejecutarlo sobre Random Forest sin modificar el mecanismo central.

**Si Random Forest es más sensible al vector de referencia, y su intento de corregirlo falló, ¿su framework realmente funciona bien sobre Random Forest?**

Sí funciona, en el sentido de que produce atribuciones estadísticamente significativas y coherentes con el comportamiento real de Random Forest. Es más sensible al vector de referencia que los modelos neuronales, y reporto eso explícitamente en vez de ocultarlo. Además expliqué mecánicamente por qué ocurre: con trescientas celdas aplanadas y árboles de profundidad acotada, la mayoría de las celdas ocluidas no participa en el camino de decisión de la mayoría de los árboles, sin importar qué valor de referencia se use ahí.

**¿Por qué confiar en resultados donde ningún modelo supera la persistencia ingenua?**

Porque el objetivo de mi framework es explicar el comportamiento del modelo entrenado, sea este bueno o malo prediciendo. Ese hallazgo no es una falla de mi propuesta, es evidencia de honestidad científica, y confirma que la tabla de métricas original medía en gran parte autocorrelación trivial del precio, no capacidad predictiva real.

**¿Por qué evaluó solamente el S&P 500?**

Es la limitación de alcance principal que reconozco en mi trabajo. El código ya soporta cambiar de índice financiero modificando un único parámetro de configuración, y replicar el pipeline sobre otro índice, como el NASDAQ Composite, es el primer punto de mi trabajo futuro.

**¿Cuál es su contribución real, en una frase?**

Un mecanismo de explicación temporal que es simultáneamente agnóstico al modelo, sin necesidad de gradientes, y de costo fijo por instancia, algo que ningún trabajo previo, incluyendo el linaje completo de DynaMask cinco años después de su publicación original, logra al mismo tiempo.
