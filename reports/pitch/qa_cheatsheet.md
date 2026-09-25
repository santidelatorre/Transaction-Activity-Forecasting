# Defensa ante el jurado

| Pregunta | Respuesta defendible |
| --- | --- |
| ¿Qué modelo está integrado? | V3-A, candidato A de `V3Model`; 75% modelo numérico con identidad familiar y 25% heurística de periodicidad. Producto V4 no significa modelo V4. Base SHA `051ce64a7cf0ab999f8aacb81fa405d5fa0257cf`. |
| ¿Resultado medido? | Macro-F1 VALID 0.42411109773651157, accuracy 0.461. Medición local reproducida. No tenemos score TEST etiquetado ni prometemos 0.8. |
| ¿Leakage? | Historial estrictamente anterior a 2026-01-01, particiones por clientes disjuntos, IDs para agrupar/unir y no como features. El mapeo supervisado familiar usa cinco folds internos al construir features de entrenamiento. Inferencia rechaza clientes usados en fit. Esto reduce vías concretas de leakage, no demuestra ausencia universal. |
| ¿Reutilizaron VALID? | Sí. VALID influyó en decisiones del equipo y promoción de baseline; su resultado no es independiente. La demo no selecciona otra receta. Para medir generalización hace falta un holdout nuevo no estudiado. |
| ¿Por qué entrenar con TRAIN+VALID? | Es la receta de submission congelada. La métrica mostrada procede de un modelo ajustado sólo con TRAIN; el predictor de TEST se reajusta con TRAIN+VALID. No presentamos el entrenamiento final como otra validación. |
| ¿Usaron etiquetas TEST? | No están disponibles ni se leen. Los ejemplos se eligen por margen de scores e historial previo al corte. La selección de ejemplos no mide éxito; nunca se comprueba si acertaron usando TEST labels. |
| ¿Qué significa none? | Clase oficial: ninguna familia recurrente predicha dentro del horizonte. No significa baja confianza, abstención, fallo del API o garantía de ausencia de pagos. |
| ¿Qué pasa con el shift? | La degradación de identidad se ve en descripciones genéricas y familias desconocidas para el mapa. Mostramos esos indicadores por cliente, pero no afirmamos disponer de un detector validado de shift ni garantía de transferencia a otro banco. |
| ¿Por qué un agente? | Decide cuánto investigar y qué herramientas consultar según margen, ambigüedad, identidad, desacuerdo y evidencia. Deja una traza y una condición de parada. Hace revisable la incertidumbre; no mejora el Macro-F1 por sí mismo. |
| ¿Es un LLM? | No en esta entrega. Es un router determinista condicional con ocho herramientas permitidas y máximo ocho pasos. `DecisionPolicy` permite un backend de razonamiento; la traza siempre usa resultados reales y la conclusión no acepta hechos libres generados por el modelo. |
| ¿Por qué no LLM-only? | El predictor real se puede reproducir y medir con el contrato oficial. Un LLM no añade evidencia por imaginar identidades o fechas. Lo usaríamos para decidir consultas, manteniendo la predicción y las herramientas verificables. No hemos medido una comparación LLM-only. |
| ¿Las cifras son probabilidades? | No. Se muestran como scores sin calibrar. La diferencia entre los dos scores mayores orienta la investigación; no es una probabilidad de acierto. |
| ¿Cómo calculan la recurrencia? | Pagos de tarjeta salientes agrupados por descripción normalizada y moneda, con observaciones, mediana de intervalos, desviación de intervalos y variabilidad de importe. La regla de apoyo exige al menos tres observaciones, regularidad ≥0.75 y CV de importe ≤0.15; es descriptiva, no una etiqueta real de suscripción. |
| ¿Conocen la identidad real del comercio? | No necesariamente. Las asociaciones familiares se aprenden de targets por cliente, no de labels históricos por transacción. Mostramos esa procedencia; una familia asociada no verifica una identidad. |
| ¿Es una explicación causal? | No. Mostramos scores y contexto histórico calculable. No calculamos SHAP ni contribuciones individuales en esta demo y no afirmamos que una transacción concreta causó la decisión. |
| ¿Usaron unlabeled data? | La baseline V3-A integrada y esta demo no usan el archivo unlabeled. Hay investigación histórica del equipo sobre él; no atribuimos sus resultados a este modelo. |
| ¿Generaliza a clientes nuevos? | Se evalúa en clientes disjuntos del ajuste, dentro del dataset del reto. No equivale a demostrar rendimiento en datos bancarios reales, comercios nuevos o una distribución diferente. |
| ¿El oracle es el score del modelo? | No. Los oracles históricos usan información privilegiada para diagnosticar límites. Ningún oracle se muestra como rendimiento desplegable ni entra en la predicción de esta demo. |
| ¿Puede el agente cambiar la entrega? | No hay herramienta de escritura ni endpoint de submission. El comando de entrega importa el predictor directamente, valida las 1.000 filas y registra su procedencia. La prueba real comprueba que investigar no modifica los artefactos. |
| ¿Cómo reproducir? | Seguir `DEMO_README.md`: dependencias fijadas, datos oficiales locales, preparación del predictor, build de React, Uvicorn y healthcheck. Lock de fuentes, hashes de datos, versiones y hashes de artefactos acompañan cada bundle. |
| ¿Qué falta? | Calibración, holdout independiente, identidades verificadas, validación con otro dominio, autenticación para despliegue externo y backend LLM conectado. La demo es local y no ejecuta acciones bancarias. |

Frase ante una duda no medida: «No lo hemos medido; podemos mostrar exactamente
qué datos y código sostienen esta afirmación y qué experimento faltaría». Nunca
convertir incertidumbre en una cifra, identidad, fecha o importe inventados.
