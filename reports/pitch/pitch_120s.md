# Pitch 120 segundos

Un banco ve transacciones. Su cliente quiere saber qué compromisos podrían
volver a aparecer. La distancia entre esas dos cosas es el problema que
resolvemos: predecir la próxima familia recurrente dentro de 90 días y mostrar
qué evidencia permite interpretarla.

Las transacciones son difíciles porque una misma descripción puede agrupar
comercios distintos, una identidad puede aparecer fragmentada y un pago repetido
no demuestra por sí solo una suscripción. Descripciones como «service payment»
ocultan precisamente la identidad que querríamos conocer.

Nuestro predictor congelado es V3-A: CatBoost con agregados históricos y
asociaciones de familia, combinado con una heurística de periodicidad en una
mezcla fija 75/25. No entrenamos otro modelo para esta demo. Reproducimos
**Macro-F1 0,4241** en las ocho clases; la accuracy es **0,461**. Son métricas
distintas. VALID fue reutilizado durante el desarrollo, así que no afirmamos
haber medido una generalización independiente.

En el cliente C002229, la familia predicha es mobile y encontramos recurrencia
histórica de apoyo. Aun así, la pantalla conserva la degradación de identidad.
Ahora cambiamos a C000796: music y otra alternativa están casi empatadas.
El agente inspecciona la calidad de datos, la recurrencia, el historial y las
alternativas. La secuencia depende de los hallazgos; no es un recorrido fijo.
El caso claro utiliza tres herramientas y el ambiguo seis.

Hoy el agente funciona sin API externa mediante una política condicional.
Su interfaz permite conectar un modelo de razonamiento, pero todas sus
herramientas son de lectura y tienen un límite de pasos. Si falta evidencia,
lo declara; no inventa un comercio ni cambia la familia oficial.

El valor es una conversación mejor informada: revisar con el cliente un posible
compromiso recurrente, con hechos, alternativas y límites visibles. No mostramos
saldos, fechas garantizadas ni importes futuros exactos. Conservamos el SHA del
modelo y un comando separado genera y valida la submission de 1.000 clientes.
