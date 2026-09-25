# Pitch 60 segundos

¿Cuál será el próximo compromiso recurrente de un cliente? El historial parece
contener la respuesta, pero las descripciones genéricas y los comercios ambiguos
hacen difícil distinguir una suscripción de una compra aislada.

Recurring Insights convierte esa incertidumbre en una conversación verificable.
Nuestro modelo real, V3-A, combina identidad de familia, historial y una
heurística de periodicidad para predecir una de ocho clases en 90 días.
Reproducimos un Macro-F1 de **0,4241** en VALID, reutilizado durante el desarrollo;
no lo presentamos como rendimiento garantizado.

Aquí vemos un caso claro y otro con alternativas casi empatadas. El agente
decide qué investigar: calidad de identidad, recurrencia o alternativas.
Muestra hechos calculados y se detiene cuando agota la evidencia. Hoy usa una
política condicional determinista, preparada para conectar un modelo de razonamiento.

El cliente y el banco pueden revisar posibles compromisos con contexto y
límites visibles. No prometemos fechas ni importes futuros. La investigación
nunca modifica la predicción oficial; modelo, demo y submission son reproducibles.

Guion visual: 0–15 s predicción; 15–30 s evidencia del caso claro;
30–48 s caso ambiguo e investigación; 48–60 s conclusión y límite.
