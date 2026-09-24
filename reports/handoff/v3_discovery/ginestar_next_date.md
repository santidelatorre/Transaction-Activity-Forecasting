# Ginestar / Carles: forecasting de la próxima ocurrencia

Experimento en `discovery/ginestar-next-date`, base congelada
`5884ddd07a19b2e19b44b4fd55afdf4eb6dcf749`. No se modifica el modelo V2 de producción.

## Diseño y límites de la evidencia

Solo se usan transacciones de train para seleccionar el método. El historial observado
abarca 2024-11-07 a 2025-12-31 y contiene 147.459 transacciones de 2.000 clientes.
No se leen labels de train en la investigación temporal; se usan únicamente en el
diagnóstico supervisado opcional de V2. No se lee test. El loader rechaza timestamps
iguales o posteriores al cutoff oficial.

Los pseudo-cutoffs son 2025-04-01 y 2025-07-01 para desarrollo/selección y
2025-10-01 para evaluación temporal reservada. Sus ventanas de 90 días no se solapan
y están completas antes de 2026-01-01. Se reutilizan clientes a través del tiempo:
esto mide generalización temporal, no generalización a clientes nuevos. No se utilizan
los resultados de octubre para cambiar el método seleccionado.

Un stream es `client_id + description`, con la normalización existente a minúsculas
y sin espacios exteriores; no hay clustering ni clasificación de familias. Se agrupan
eventos por día UTC, porque el objetivo es una fecha, no una hora. Se exigen tres
días distintos, un span de al menos 14 días y mediana de gaps >=3 días, todo calculado
estrictamente antes de cada cutoff. No se exige que el stream reaparezca para admitirlo.
La descripción y el ID son claves de agrupación, nunca predictores numéricos ni reglas
de desempate. El conjunto incluye streams irregulares: repetición no certifica una
suscripción real. Los resultados no identifican por sí solos el cuello de botella de
clasificación de familias.

La verdad de próxima fecha es el primer día de ese stream en `[cutoff, cutoff+90d)`.
MAE, mediana de error y tolerancias se calculan solo sobre eventos observados dentro
de esa ventana; no se imputa una fecha a los demás. Recall y precisión de ocurrencia
usan todos los candidatos. «None temporal» significa ausencia de eventos de los
candidatos elegibles, no ausencia de cualquier transacción ni la clase oficial `none`.
La observabilidad hasta fin de 2025 se presupone a nivel del dataset; no hay indicador
individual de cierre de cuenta para distinguir churn de censura de captura.

Para ranking se exigen >=2 candidatos y al menos uno con evento en 90 días. Si varios
streams ocurren el primer día, cualquiera cuenta como correcto. Los empates de fecha
predicha se puntúan con el crédito esperado de un orden uniforme, sin usar IDs.
Top-2 mide si uno de los dos primeros pertenece al conjunto de primeros reales.
Los pronósticos fuera del horizonte se ordenan al final; si todos quedan fuera,
top-1/top-2 son cero. El error del stream elegido es el promedio por cliente del error
de sus candidatos empatados observables; se reporta aparte su fracción sin evento
observable. Para los pocos casos de abstención total, este error es el del ranking
hipotético, no de una acción emitida. No representa una pérdida penalizada por churn.

## Métodos y congelación

Se comparan doce recetas, sin grid de hiperparámetros:

| Método | Regla |
| --- | --- |
| last_gap | Último gap |
| mean | Media de todos los gaps |
| median | Mediana de gaps |
| trimmed_mean | Media después de recortar floor(20% × número de gaps) por extremo |
| ewma | EWMA de gaps, alpha=0,4, inicializada en el primer gap |
| median_phase | Retícula de periodo mediano, fase robusta por mediana de residuos, permite ciclos omitidos |
| weekly | Misma estimación de fase con periodo fijo de 7 días |
| fortnightly | Periodo fijo de 14 días (quincenal operacional) |
| monthly | Día del mes mediano; fin de mes si >=60% del historial cae en fin de mes |
| quarterly | Mismo anclaje de calendario, incrementos de tres meses |
| automatic | Selecciona weekly/fortnightly/monthly/quarterly por proximidad del gap mediano; fallback median_phase |
| automatic_active | automatic con veto de ocurrencia si la antigüedad supera 2,5 periodos medianos |

Las cinco reglas de gaps predicen inmediatamente en el cutoff si su fecha quedó
vencida. Las reglas de fase/calendario avanzan hasta la siguiente ocurrencia proyectada.
Mensual y trimestral usan meses reales, conservan el día ancla al saltar meses y
recortan al último día del mes cuando es necesario; no sustituyen un mes por 30 días.
La regla trimestral toma el mes del último evento como referencia de fase.

Para la selección automática, el periodo mediano debe estar a <=25% de 7, 14,
30,4375 o 91,3125 días y MAD(gaps)/mediana <=0,35. En otro caso se etiqueta
`irregular` o `other` y se usa median_phase. Estas etiquetas son estimaciones del
pasado, no ground truth de periodicidad. Las tolerancias se fijaron antes de evaluar.

Se selecciona por mayor top-1 agregado en abril+julio; los desempates usan menor
MAE y luego nombre de método. El ganador se registra en `frozen_method.json` antes
de cargar el historial de validation y antes de cualquier acceso a sus labels.
No se reajusta por los resultados posteriores. Los parámetros y el hash SHA-256
del fichero de train quedan en ese manifiesto.

## Resultados y selección

| Pseudo-cutoff | Uso | Streams | Clientes con >=1 candidato | Eventos de stream en 90 días |
| --- | --- | --- | --- | --- |
| 2025-04-01 | Desarrollo | 3.717 | 1.512 | 2.248 |
| 2025-07-01 | Desarrollo | 9.801 | 1.901 | 5.804 |
| 2025-10-01 | Holdout temporal | 16.872 | 1.982 | 9.578 |

Son 30.390 pares stream/cutoff, no 30.390 streams distintos. Se excluyen del
ranking los clientes sin dos candidatos y de su denominador los casos None.
Octubre tiene 1.938 clientes con >=2 candidatos: 1.867 casos de ranking y 71 None.
El crecimiento de candidatos explica parte de la caída de top-1 entre fechas;
no debe interpretarse como deterioro puro de la señal temporal.

El método congelado es **automatic**, elegido por ranking de desarrollo:

| method | mae | top1 | top2 |
| --- | --- | --- | --- |
| automatic | 27.729 | 28.00% | 55.72% |
| automatic_active | 27.729 | 27.18% | 55.20% |
| ewma | 31.340 | 27.04% | 53.58% |
| fortnightly | 31.412 | 26.85% | 52.40% |
| last_gap | 35.389 | 26.42% | 53.74% |
| mean | 31.195 | 26.77% | 53.72% |
| median | 31.752 | 26.95% | 54.11% |
| median_phase | 27.666 | 27.43% | 55.43% |
| monthly | 26.414 | 27.03% | 53.21% |
| quarterly | 34.413 | 25.96% | 52.28% |
| trimmed_mean | 31.288 | 26.72% | 53.86% |
| weekly | 34.215 | 27.78% | 53.11% |

### Fechas: holdout de octubre

Todos los métodos comparten los mismos 16.872 candidatos y 9.578 eventos evaluables.
MAE y median_ae están en días. Los porcentajes de tolerancia tienen como denominador
los eventos evaluables; recall/precision_90d evalúan presencia en horizonte.

| method | mae | median_ae | within_3d | within_7d | within_14d | recall_90d | precision_90d |
| --- | --- | --- | --- | --- | --- | --- | --- |
| automatic | 29.669 | 24.000 | 9.67% | 19.18% | 33.62% | 94.46% | 57.22% |
| automatic_active | 29.669 | 24.000 | 9.67% | 19.18% | 33.62% | 75.52% | 57.53% |
| ewma | 33.747 | 28.302 | 7.32% | 15.31% | 28.05% | 93.74% | 57.13% |
| fortnightly | 30.505 | 25.500 | 11.83% | 21.86% | 35.02% | 100.00% | 56.77% |
| last_gap | 39.838 | 32.000 | 7.63% | 15.58% | 26.74% | 88.56% | 57.03% |
| mean | 33.323 | 28.000 | 7.65% | 15.61% | 28.06% | 94.33% | 57.25% |
| median | 33.752 | 28.500 | 8.56% | 16.65% | 28.96% | 94.12% | 57.20% |
| median_phase | 29.667 | 24.000 | 9.09% | 18.96% | 33.39% | 93.69% | 57.24% |
| monthly | 25.577 | 19.000 | 12.65% | 24.84% | 42.15% | 100.00% | 56.77% |
| quarterly | 33.101 | 31.000 | 6.91% | 14.78% | 27.04% | 99.68% | 56.74% |
| trimmed_mean | 33.494 | 28.500 | 7.89% | 15.79% | 28.16% | 94.33% | 57.25% |
| weekly | 33.341 | 29.000 | 10.33% | 18.92% | 29.56% | 100.00% | 56.77% |

### Ranking entre streams: holdout de octubre

| method | top1 | top2 | chosen_mae | chosen_censored |
| --- | --- | --- | --- | --- |
| automatic | 16.32% | 30.34% | 31.267 | 41.96% |
| automatic_active | 16.22% | 29.52% | 27.714 | 40.09% |
| ewma | 13.91% | 27.91% | 36.644 | 44.45% |
| fortnightly | 12.98% | 26.29% | 35.139 | 44.81% |
| last_gap | 14.44% | 28.37% | 36.184 | 44.29% |
| mean | 13.93% | 27.62% | 36.624 | 44.43% |
| median | 13.56% | 27.43% | 36.962 | 44.93% |
| median_phase | 16.43% | 30.77% | 30.892 | 42.39% |
| monthly | 15.42% | 31.03% | 30.755 | 44.50% |
| quarterly | 13.39% | 27.15% | 26.643 | 42.49% |
| trimmed_mean | 13.79% | 27.72% | 36.797 | 44.48% |
| weekly | 13.61% | 26.59% | 35.572 | 45.92% |

Referencia uniforme aleatoria entre candidatos, con los mismos empates reales:
top-1 **14.08%**, top-2 **28.08%**. Automatic alcanza
**16.32% / 30.34%**: +2.24 puntos de top-1
frente al azar y +2.76 puntos frente
a mediana. El intervalo bootstrap pareado por cliente de esta última diferencia
es [1.17, 4.39] puntos (2.000 remuestras, seed=42, octubre; descriptivo,
sin corregir por comparaciones múltiples).

Median_phase supera ligeramente a automatic en top-1 de octubre, y monthly gana
en MAE. Ninguno sustituye al ganador fijado en desarrollo. La mediana con fase
reduce el MAE de 33,752 a 29,667 días; automatic obtiene 29,669. El calendario
mensual universal baja el MAE a 25,577, pero no es el mejor ranking. Esto ilustra
que minimizar el error de cada fecha y acertar el siguiente stream no son el mismo
objetivo. El 41,96% de masa de selección de automatic recae en streams sin evento
observable en 90 días: el error elegido de 31,267 días es condicional, no una
evaluación completa del daño de elegir un stream inactivo.

### Segmentos de periodicidad

La segmentación de fechas usa la periodicidad estimada del propio stream:

| periodicity | n | events_90d | mae | median_ae | within_7d | recall_90d |
| --- | --- | --- | --- | --- | --- | --- |
| fortnightly | 239 | 164 | 24.351 | 16.000 | 28.66% | 100.00% |
| irregular | 10307 | 5834 | 30.259 | 25.000 | 17.42% | 93.91% |
| monthly | 1698 | 1000 | 21.589 | 14.000 | 36.50% | 100.00% |
| other | 2941 | 1664 | 31.062 | 24.500 | 16.83% | 89.54% |
| quarterly | 1644 | 889 | 32.875 | 30.000 | 14.40% | 99.78% |
| weekly | 43 | 27 | 42.278 | 46.500 | 3.70% | 100.00% |

La segmentación de ranking usa la periodicidad estimada del stream realmente primero,
no la del elegido; `mixed` indica primeros simultáneos de periodicidades diferentes:

| periodicity | ranking_cases | top1 | top2 | random_top1 | chosen_mae |
| --- | --- | --- | --- | --- | --- |
| fortnightly | 45 | 50.00% | 72.22% | 13.98% | 18.132 |
| irregular | 1072 | 16.31% | 30.18% | 13.59% | 32.105 |
| mixed | 39 | 29.49% | 47.44% | 21.00% | 21.862 |
| monthly | 232 | 20.83% | 44.83% | 15.18% | 29.322 |
| other | 290 | 12.07% | 20.69% | 13.32% | 32.252 |
| quarterly | 185 | 4.59% | 12.97% | 14.84% | 34.937 |
| weekly | 4 | 100.00% | 100.00% | 35.42% | 26.500 |

El calendario mensual es el segmento más convincente en fechas: MAE 21,589 y
36,50% dentro de ±7 días. Quincenal mejora el ranking sobre azar, con solo 45 casos.
Semanal tiene únicamente cuatro casos de ranking: el 100% no es evidencia robusta
y convive con MAE 42,278 sobre 27 eventos. Trimestral tiene top-1 4,59%, por debajo
del azar del segmento; tres eventos y un gap cercano a un trimestre no establecen
una fase estable. El 61,09% de candidatos de octubre se etiqueta irregular.

### Jitter temporal

Jitter = MAD de gaps del pasado, en días: low <=2; medium >2 y <=7; high >7.
No es una estimación causal de ruido puro: mezcla jitter, ciclos omitidos, cambios
de frecuencia y streams no periódicos.

| jitter | n | events_90d | mae | median_ae | within_3d | within_7d | within_14d |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high | 13447 | 7574 | 30.547 | 25.000 | 7.94% | 17.07% | 31.56% |
| low | 1183 | 706 | 24.559 | 19.000 | 19.97% | 31.59% | 43.34% |
| medium | 2242 | 1298 | 27.324 | 21.250 | 14.18% | 24.73% | 40.37% |

El 79,70% de los candidatos tiene MAD >7 días. Low mejora MAE (24,559 frente
a 30,547 en high) y cobertura ±7 días (31,59% frente a 17,07%), pero tampoco da
fechas exactas de forma fiable. La media recortada apenas mejora respecto a media;
con dos o tres gaps ni siquiera recorta observaciones. La corrección de fase ayuda
más que suavizar gaps, sin resolver los cambios de actividad. No se han añadido
correcciones de festivos o días laborables después de ver estos resultados.

### None temporal

En clientes con >=2 candidatos, automatic detecta **0 de 71** None reales,
con una sola predicción None, que es falsa. Su accuracy de 96,28% refleja el
desequilibrio y es peor que predecir siempre actividad (96,34%). Automatic_active
detecta 1 de 71 (recall 1,41%, precisión 14,29%) y reduce el recall de eventos de
94,46% a 75,52%. La abstención por antigüedad no resuelve el problema.

Incluyendo clientes con un solo candidato (los 1.982 con >=1 candidato), se obtiene:

| method | n_clients_cutoffs | none_cases | none_predicted | none_recall | none_precision | none_specificity | none_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| automatic | 1982 | 99 | 3 | 1.01% | 33.33% | 99.89% | 94.95% |

Los 18 clientes de train sin candidatos en octubre quedan fuera de esta tabla;
«ningún candidato ocurre» sería verdadero por construcción y no mediría capacidad
predictiva. Tampoco se cuenta como error que la próxima transacción pertenezca a un
stream excluido: el ranking es estrictamente entre candidatos elegibles.

### Ejemplos concretos

Ejemplos anónimos del cutoff 2025-10-01, seleccionados para ilustrar mecanismos,
no como muestra representativa. Sus valores proceden de los pronósticos locales:

| Caso | Tipo estimado | Días históricos | Último evento | Gap mediano / MAD | Predicción | Real | Error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | monthly | 8 | 2025-09-19 | 25 / 8 días | 2025-10-14 | 2025-10-14 | 0 días |
| B | monthly | 7 | 2025-09-20 | 29,5 / 2 días | 2025-10-19 | 2025-10-19 | 0 días |
| C | quarterly | 3 | 2025-09-26 | 81 / 18 días | 2025-12-25 | 2025-10-01 | 85 días |
| D | quarterly | 3 | 2025-06-24 | 68,5 / 7,5 días | 2025-12-24 | 2025-10-01 | 84 días |
| E | irregular | 3 | 2025-09-29 | 158 / 69 días | 2026-03-06 | 2025-10-10 | 147 días |

El día mensual es robusto frente a desplazamientos del último evento en A/B.
C/D muestran que gaps aproximadamente trimestrales pueden producir una falsa
confianza de fase. E muestra un stream con dos gaps largos y desiguales que vuelve
mucho antes de lo proyectado. La fecha predicha de 2026 es un output; no se usó
ningún timestamp observado posterior al cutoff oficial.

### Aplicación oficial y ablación V2

Tras congelar automatic se proyectaron **10,164 streams** de
**992 clientes de validation** al cutoff 2026-01-01.
No hay ground truth de fecha post-cutoff en estos ficheros: estas proyecciones no
tienen MAE oficial medible. No se cargó test.

La ablación vuelve a entrenar la receta V2 fija con train y sus labels, conserva
CatBoost, su matriz de features, mapeo de descripciones, scores base, soporte,
sesgo None, temperatura y mezcla 75/25. Solo reemplaza, en streams elegibles,
el máximo de cercanía a ciclos usado por el factor de periodicidad por
`occurs * exp(-days_until_prediction / 90)`. El factor sigue siendo
`0.5 + support * temporal_closeness`; los streams no elegibles mantienen V2.
No se altera el código de producción ni se buscan pesos.

| Variante | Macro-F1 oficial | Accuracy |
| --- | --- | --- |
| V2 congelada, control reproducido | 0.391549456 | 0.4240 |
| V2, sustitución temporal única | 0.395253013 | 0.4270 |
| Delta | +0.003703557 | +0.0030 |

Este resultado es diagnóstico sobre validation reutilizada históricamente por el
equipo, no una estimación independiente. Se consultaron sus labels solo después
de fijar las dos predicciones; no se cambió el método ni la fórmula tras puntuarlas.
La ablación cambia una ponderación temporal agregada, no convierte V2 en un
clasificador que elija directamente la familia del stream más temprano.

### Recomendación

Conservar el experimento y su evaluación causal como herramienta de diagnóstico.
La fase aporta una mejora pequeña y medible de ranking, especialmente mensual y
quincenal, pero la exactitud absoluta, la detección de inactividad y el resultado
trimestral impiden tratarlo como solución suficiente. No promover esta fórmula
a V2 solo por una mejora frente a la mediana. El siguiente experimento debería
separar probabilidad de reaparición y fecha condicional, con nueva validación
temporal reservada; no ajustar umbrales sobre este octubre ni sobre validation.
La investigación de familias sigue siendo una línea independiente.



## Reproducción y entrega

Desde la raíz del repositorio, en PowerShell:

```powershell
$env:PYTHONPATH='src;.'
.venv/Scripts/python.exe scripts/experiments/ginestar_next_date.py --v2-diagnostic
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check .
$env:PRE_COMMIT_HOME=Join-Path (Get-Location).Path '.pre-commit-cache-ginestar-next-date'
.venv/Scripts/python.exe -m pre_commit run --all-files
```

El código reusable está en `src/transaction_forecasting/ubs/next_date.py` y los
tests en `tests/test_next_date.py`. Los outputs permanecen locales e ignorados en
`outputs/metrics/v3_discovery/ginestar_next_date/`: métricas agregadas y segmentadas,
pronósticos por pseudo-cutoff, rankings por cliente, None de todos los candidatos,
bootstrap pareado, ejemplos, manifiesto de congelación, proyecciones oficiales y
diagnóstico V2. No se versionan datasets ni predicciones individuales.

Las pruebas cubren frontera estricta del cutoff, independencia de features respecto
al futuro, febrero/fin de mes/calendario trimestral, veto de inactividad, censura,
empates predichos y simultaneidad real. El primer pytest encontró restricciones de
acceso al directorio temporal; la ejecución con permisos apropiados pasó. La caché
pre-commit anterior estaba incompleta y se creó una específica de este experimento.

Verificación final: 85 tests pasan; Ruff check, Ruff format y pre-commit pasan. Pytest emite únicamente un aviso de permisos al escribir su caché.

**TEMPORAL RANKING HELPS BUT IS NOT SUFFICIENT**
