# Guion de exposición, Capítulos 1 a 3 (Prof. Hinojosa)

Cada bloque corresponde a una diapositiva de `diapos_cap1_3.tex`. El orden sigue el esquema pedido: presentación, Capítulo 1, Capítulo 2 y Capítulo 3. Duración total estimada de 18 minutos, con más peso en el Marco Teórico que en el Estado del Arte, dentro del rango de 15 a 20. Tiempo sugerido entre paréntesis.

## Presentación (0:35)

**1. Portada (0:20).** Buenos días. Soy Rushell Zavalaga y presento el avance de mi tesis, un framework de explicabilidad agnóstico al modelo para series temporales financieras, con los tres primeros capítulos.

**2. Contenido (0:15).** Presento la introducción, el marco teórico y el estado del arte, este último solo con trabajos de 2021 a 2026.

## Capítulo 1. Introducción (6:30)

**3. La idea de la tesis en un ejemplo (0:45).** Un modelo recibe los últimos 20 días del S&P 500, un índice de las 500 mayores empresas de Estados Unidos, y predice el día siguiente. Cada día tiene 15 datos, entre precios, volumen e indicadores. Si el modelo dice que mañana sube, no dice por qué, es una caja negra. Mi tesis construye un método que, para cada predicción, indica qué dato influyó y de qué día, por ejemplo el volumen de hace tres días. El resultado es una tabla de importancias, un valor por día y por dato.

**4. Vocabulario financiero y datos (0:40).** Esta tabla fija solo el vocabulario financiero. Un portafolio es el conjunto de inversiones de una institución. Cada día tiene precios de apertura, cierre, máximo y mínimo, y el volumen negociado. Los indicadores técnicos son cálculos sobre precios pasados, como el promedio de 5 días o el RSI. El retorno logarítmico es la variación relativa entre dos días. Un mercado bajista es un tramo donde el precio cae, como 2022, y alcista donde sube, como 2023 y 2024. La entrada del modelo es una ventana de 20 días por 15 datos.

**5. Contexto (1:00).** Estas predicciones se usan para decidir sobre un portafolio, cuándo comprar, mantener o vender. Hay dos riesgos. El de mercado es perder porque el precio se movió en contra, aunque el modelo se entendiera. El de modelo es perder porque se decidió con un modelo incorrecto o mal usado, y lo define el SR 11-7, una guía de la Reserva Federal de Estados Unidos que exige a los bancos gestionar sus modelos. Cohen y colaboradores muestran que la opacidad lo agrava, porque no se sabe si el modelo aprendió una relación real o una coincidencia hasta que la decisión ya se tomó. Mi tesis se ubica en el riesgo de modelo.

**6. Motivación (0:45).** El aprendizaje profundo reduce el error frente a los métodos clásicos, pero no explica sus resultados, y la regulación ya lo exige. El SR 11-7 pide entender los límites del modelo antes de usarlo, y el Reglamento de IA de la Unión Europea exige transparencia en usos como el crédito. Sovrano y colaboradores muestran que los métodos actuales aún no cumplen. Un modelo que no se explica no se puede auditar. Mi pregunta es qué variable fue relevante y cuándo lo fue.

**7. Planteamiento del problema (0:55).** El problema es que los modelos que predicen series financieras no explican qué dato de qué día influyó en cada predicción. Se necesita un método con tres propiedades. Que funcione con cualquier modelo sin acceder a su interior, que respete el orden temporal de los datos, y que tenga un costo fijo por predicción, sin optimizar nada cada vez. Hoy ninguno cumple las tres. Los métodos que sirven con cualquier modelo tratan cada día como un dato suelto y rompen el orden temporal. Los que sí respetan el orden necesitan el gradiente del modelo, así que no sirven para Random Forest. Y ninguno se validó en series financieras. Los métodos concretos los detallo en el Capítulo 3.

**8. Justificación (0:45).** Hay cuatro razones. El costo fijo, porque perturbar una celda, un dato en un día, no depende de otra, y todas caben en un solo batch, mientras que los métodos con gradiente son secuenciales. Cierra la brecha, porque perturba una celda a la vez con el orden intacto y sin gradientes. Random Forest es competitivo en datos tabulares y esta línea lo excluye. Y la responsabilidad, porque reporto también los resultados desfavorables.

**9. Objetivo general (0:20).** Desarrollar un framework agnóstico que atribuya importancia por variable y paso de tiempo, sin gradientes ni acceso al interior del modelo, con costo fijo por predicción.

**10. Pipeline (0:40).** El flujo tiene seis etapas. Datos, preparación en ventanas, modelo, interfaz uniforme de predicción, motor de perturbación y, como salida, la matriz de importancia. Las etapas cuatro y cinco, en naranja, son el núcleo agnóstico de la propuesta.

**11. Objetivos específicos (0:40).** Son cinco. Diseñar la interfaz uniforme de predicción. Formular el motor de perturbaciones por celda. Entrenar LSTM, Transformer y Random Forest sobre el S&P 500. Verificar que las explicaciones reflejan el comportamiento de cada modelo. Y evaluar frente al estado del arte agnóstico.

## Capítulo 2. Marco Teórico (6:10)

**12. Serie temporal y ventana (0:45).** Una serie temporal multivariada reúne varias variables medidas cada día. El modelo no recibe la serie completa, recibe una ventana de T días con V variables, que en mi caso es una tabla de 20 por 15. Con esa tabla, el modelo predice el valor del día siguiente. Cada celda de la tabla, una variable en un día concreto, por ejemplo el volumen de hace tres días, es la unidad sobre la que atribuyo importancia. Todo el trabajo gira alrededor de esa idea, saber cuánto pesa cada celda en la predicción.

**13. Modelos (0:45).** Uso tres modelos que procesan la ventana de forma distinta, para comprobar que el método funciona con cualquiera. La LSTM la recorre paso a paso. El Transformer compara todos los días entre sí con autoatención. Random Forest promedia árboles de decisión sobre la ventana aplanada, sin noción de secuencia. Los dos primeros son redes diferenciables, tienen gradiente, y el tercero no. Esa diferencia decide qué métodos de explicación se pueden usar con cada uno. Además, Random Forest no extrapola fuera del rango que vio al entrenar, por eso trabajo con retornos y no con precios.

**14. Explicabilidad y agnosticismo (1:00).** Mi trabajo produce explicaciones locales, es decir, sobre una predicción concreta, y post-hoc, sobre modelos que ya están entrenados. Un método es agnóstico cuando trata al modelo como una caja negra que solo se consulta, le da una entrada y observa la salida. Pero ser agnóstico por diseño no basta. Adebayo y colaboradores mostraron que varios métodos dan explicaciones casi idénticas incluso con los pesos del modelo aleatorizados, es decir, explican el método y no el modelo. Y Rudin advierte que una explicación post-hoc no es completamente fiel. Por eso en este trabajo la fidelidad se mide y no se supone. Mi criterio es que las explicaciones deben cambiar entre modelos de naturaleza distinta.

**15. Familias de atribución (0:50).** Hay tres familias. Las basadas en gradientes usan la derivada de la salida respecto de la entrada, y por eso requieren un modelo diferenciable, no aplican a Random Forest. Las basadas en atención leen los pesos de atención como importancia, dependen de la arquitectura, y Jain y Wallace muestran que esos pesos pueden cambiar sin alterar la predicción. Las basadas en perturbación modifican parte de la entrada y observan cómo cambia la salida. Solo consultan el modelo, por eso son agnósticas, y es la familia de mi propuesta. Su costo es que necesitan varias consultas por explicación.

**16. Métodos de perturbación (1:10).** La oclusión reemplaza una celda de la ventana por un valor de referencia y mide cuánto cambia la predicción. Esa diferencia en valor absoluto es la importancia de esa celda. La referencia define qué significa ausencia de información, puede ser la media de entrenamiento o cero, y como cambiarla puede cambiar la atribución, mido su sensibilidad. Los valores de Shapley reparten la contribución entre todas las coaliciones posibles de variables. Su cálculo exacto crece de forma exponencial, por eso se aproximan por muestreo. LIME ajusta un modelo lineal local sobre perturbaciones. Ambos suponen variables independientes, lo que falla en series temporales, y la entrada perturbada puede quedar fuera de la distribución de entrenamiento.

**17. Explicabilidad temporal (0:50).** Extiende la atribución al tiempo. En lugar de un valor por variable, produce una matriz de T por V que responde qué variable influyó y cuándo. Ismail y colaboradores muestran que aplicar métodos pensados para imágenes o texto a series temporales mezcla el eje del tiempo con el de las variables. Una explicación temporal válida debe cumplir dos cosas. Respetar el orden, sin usar información futura para explicar un instante pasado. Y distinguir la dependencia entre pasos del efecto retrasado. Un método que perturba cada día por separado ignora ambos efectos.

**18. Evaluación (0:50).** Hay dos situaciones. Con verdad conocida, en datos sintéticos donde yo sé qué celdas son relevantes, comparo con precisión y cobertura. La fidelidad se mide con la curva AOPC, que elimina primero las celdas más importantes y mide cuánto cae la predicción. En series reales no hay verdad conocida, y ahí uso la correlación de Spearman entre las matrices de modelos distintos, la prueba de Mann-Whitney contra una asignación aleatoria y la estabilidad frente a la referencia, el orden y el régimen de mercado.

## Capítulo 3. Estado del Arte (5:00)

**19. Criterio de organización (0:20).** Incluyo solo trabajos de 2021 a 2026 y los ordeno por cómo construyen la explicación, en tres ejes. Perturbación agnóstica, frameworks temporales con el linaje de DynaMask, y explicadores aprendidos.

**20. Eje 1 (0:45).** Yadav y Subbian diagnostican por qué gradiente, oclusión y permutación fallan en predicción dinámica, porque romper la dependencia temporal corrompe la cadena de predicciones. ShapeX aplica Shapley sobre segmentos, pensado para clasificación. Y los trabajos financieros usan SHAP o LIME directo sobre ventanas, sin verificar que respeten el orden causal.

**21. Eje 2a (0:40).** WinIT captura efectos retrasados sin gradientes, pero solo se evaluó en modelos recurrentes y tareas clínicas. TimeSHAP está limitado a modelos secuenciales y TsSHAP es univariado. Roshinta y Szűcs, y GS-SHAP, trabajan con ventanas o grupos, y pierden resolución por variable.

**22. Eje 2b, DynaMask (1:15).** DynaMask es el trabajo más cercano, porque su salida es la misma matriz de T por V. Aprende una máscara que mezcla la entrada con una versión perturbada, y la optimiza por gradiente para cada predicción. Por eso requiere gradiente y no corre en Random Forest, su costo crece con las iteraciones, y su operador supone una serie suave, algo que choca con los cambios de régimen. ExtrMask y ContraLSP lo mejoran, pero mantienen el gradiente, y ninguno se evalúa en finanzas. Las máscaras buscan lo suficiente para mantener la predicción, y la oclusión mide lo necesario.

**23. Eje 3 (0:35).** TimeX y TimeX++ entrenan un modelo sustituto, y Zheng y colaboradores unifican atribución y contrafactual. Comparten que entrenan una red auxiliar, por lo que su costo no es el de consultar el modelo, y que se evalúan fuera de las finanzas.

**24. Síntesis (0:40).** Esta tabla resume cada trabajo y su brecha. SHAP y LIME asumen independencia, WinIT y TimeSHAP solo sirven para modelos secuenciales, DynaMask y sucesores necesitan gradiente, y los explicadores aprendidos entrenan una red. Ninguno combina los rasgos de mi propuesta.

**25. La brecha (0:45).** Ningún trabajo previo cumple a la vez tres condiciones. Agnosticismo real, sin gradientes ni estructura secuencial. Resolución por celda, sin agrupar. Y verificación empírica de que las explicaciones cambian entre modelos distintos. De ahí mi hipótesis, que es posible un método de perturbación ordenado en el tiempo y agnóstico que produzca explicaciones consistentes con cada arquitectura.

**26. Referencias.** Para consulta.

## Preguntas que conviene preparar

- **¿Por qué FIT no aparece en el estado del arte?** Es de 2020, fuera de la ventana de cinco años. Está descrito en el marco teórico como base conceptual, y WinIT, que lo extiende, sí está en el estado del arte.
- **¿Qué hiciste específico para Random Forest?** El Capítulo 4 investiga una referencia condicionada a la hoja. El resultado no confirmó la hipótesis inicial, y lo reporto así.
- **¿Qué pasó con el enfoque de mejorar DynaMask?** La tesis usa a DynaMask y a sus sucesores como el hilo del estado del arte y toma su limitación, la dependencia del gradiente, como el problema que resuelve la propuesta.
