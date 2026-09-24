# EDA de distribución y patrones de clase

## Estado de la evidencia (24-09-2026)

Los archivos de datos reales no están presentes en este checkout (`data/*` solo
contiene `.gitkeep`), así que no pude recalcular métricas. Sin embargo,
`origin/main:docs/UBS_DATASET_ANALYSIS.md` publica un análisis reproducible del
dataset UBS y `origin/main:src/transaction_forecasting/ubs/data.py` define la
tarea real: **una predicción de clase por `client_id`**, con historial anterior al
1-01-2026 y ocho etiquetas. Todas las cifras siguientes proceden de ese informe
del repositorio y quedan pendientes de verificación independiente aquí.

| Partición | Transacciones | Clientes | Mediana eventos/cliente | Primer y último evento |
| --- | ---: | ---: | ---: | --- |
| Train | 147459 | 2000 | 72 | 07-11-2024 a 31-12-2025 |
| Validation | 73898 | 1000 | 73 | 07-11-2024 a 31-12-2025 |
| Test | 75761 | 1000 | 74 | 07-11-2024 a 31-12-2025 |

Los conjuntos de clientes no se solapan, y el informe registra cero eventos a
partir del corte. Train tiene un promedio de 73,73 transacciones por cliente;
el historial medio cubre 404,59 días. El último evento del 99,5 % de los
clientes ocurre dentro de los 30 días previos al corte. No hay clientes de
historial escaso entre los agregados publicados, pero el informe no proporciona
un recuento exacto de clientes con 0, 1 o 2 eventos. No inventarlo.

| Clase | Train | Train % | Validation |
| --- | ---: | ---: | ---: |
| cloud | 190 | 9,50 | 89 |
| gym | 190 | 9,50 | 121 |
| insurance | 214 | 10,70 | 99 |
| mobile | 191 | 9,55 | 104 |
| music | 198 | 9,90 | 93 |
| software | 195 | 9,75 | 104 |
| streaming | 225 | 11,25 | 97 |
| none | 597 | 29,85 | 293 |

`none` concentra casi el 30 %; las otras clases rondan el 9,5–11,25 %. Esto
justifica revisar soporte y recall por clase además de Macro-F1. El importe
train es positivo en todas las filas; mediana 73,08, p99 7983,11 y máximo
8999,98 en **unidades de monedas mezcladas**. CHF aparece en 73086 filas,
EUR en 38146 y USD en 21957; no debe calcularse una dispersión monetaria global
sin separar moneda. No hubo filas duplicadas exactas ni pares
`client_id`/`timestamp` duplicados en train según el informe.

Hay 33051 grupos repetidos `(client_id, description)` en train y todos los
clientes tienen al menos uno. La mediana de apariciones por grupo es 3; el
30,69 % tiene desviación de intervalo de hasta tres días y el 25,22 % muestra
CV de importe de hasta 5 %. Los bins descriptivos cuentan 9338 grupos de
cadencia mensual aproximada y 796 semanal. La descripción más frecuente es
`salary` (12031); le siguen `atm withdrawal` (8083) y `fresh foods` (6649).
Hay señales directas de familias en el texto: por ejemplo, `cloud access`
aparece en el 80,53 % de clientes cloud frente al 15,91 % de los demás, y
`member plan` está 6,24 veces más presente entre clientes `none`. Son
asociaciones de train, no efectos causales ni garantía de generalización.

## Análisis que ejecutar al recibir los datos

Todos los recuentos de clase deben hacerse sobre **clientes etiquetados** si la
unidad de predicción es el cliente. Reportar también el número de transacciones
por clase, pero sin confundirlo con el soporte de la métrica. Separar train y
validation, y calcular asociaciones descriptivas solo con train. Para cada
resultado, dar recuentos absolutos, porcentajes, denominadores y percentiles;
desglosar por clase y por antigüedad del historial.

| Bloque | Medidas concretas | Posible señal para las ocho clases |
| --- | --- | --- |
| Soporte | Clientes únicos por etiqueta, cuota, faltantes, clases ausentes en cada corte temporal | Fragilidad de Macro-F1 en clases minoritarias |
| Cobertura | Transacciones por cliente: 0, 1, 2, 3-5, 6-10, >10; p10/p25/mediana/p75/p90/p99 | Separar arranque en frío de patrones maduros |
| Importes | Signo, moneda, ceros, p1/p25/mediana/p75/p99, IQR y MAD; dispersión dentro de cliente y descripción | Pagos de cuantía estable, cobros variables, cargos extraordinarios |
| Recurrencia | Días entre eventos consecutivos por cliente y por descripción/merchant; mediana, MAD, coeficiente de variación | Regularidad semanal, quincenal, mensual o irregularidad |
| Calendario | Día de semana/mes, fin de mes, ventanas de 7/14/30 días, hora si existe | Preferencias de fecha distintas entre clases |
| Texto | Descripciones más frecuentes y clientes distintos por descripción, normalizadas y sin normalizar; cobertura del top 10/50/100 | Comercios o conceptos dominantes y variantes de nombre |
| Clientes escasos | Distribución de clase y error de baseline en historiales de 0, 1 y 2 eventos | Reglas de fallback y necesidad de priors globales |
| Anomalías | Duplicados, timestamps iguales, reversos/reembolsos, importes extremos por moneda y cliente | Evitar que eventos atípicos falseen periodicidad o tendencia |

Para texto frecuente, evitar publicar descripciones que identifiquen personas
o cuentas; mostrar agregados y ejemplos anonimizados. Para periodicidad,
comparar intervalos en días con bandas tolerantes a semanas y meses reales,
incluidos meses de 28 a 31 días. Calcular proporciones por clase respecto a su
propio soporte y medir elevación frente al promedio global; un top de frecuencia
sin denominador tiende a reflejar solo las clases grandes.

## Propuestas de features a contrastar

Estas son **hipótesis**, no hallazgos empíricos ni decisiones para cambiar el
pipeline. Cada feature debe calcularse solo con eventos anteriores al instante
de predicción.

1. `history_count`, `history_span_days`, `days_since_last`: cobertura y recency;
   incluir banderas explícitas para 0, 1 y 2 transacciones.
2. `count_7d/30d/90d`, `recent_to_prior_count_ratio`: frecuencia y aceleración
   reciente con ventanas cerradas antes del corte.
3. `gap_median_days`, `gap_mad_days`, `gap_cv`, `gap_last_vs_median` por cliente
   y descripción normalizada: regularidad y siguiente evento probable.
4. `weekly_fit`, `monthly_fit`, `month_end_share`, `weekday_concentration`:
   periodicidad tolerante al calendario, con soporte mínimo y valor nulo si no
   existen intervalos suficientes.
5. `amount_median`, `amount_mad`, `amount_cv`, `last_amount_vs_median`, por
   moneda y dirección: estabilidad y cambios recientes sin mezclar divisas.
6. `description_top_share`, `description_unique_count`, `merchant_entropy`,
   `last_description_seen_before`: concentración y novedad; normalización de
   texto aprendida solo desde train cuando incluya estadísticas globales.

Priorizar el contraste por clase y por longitud de historial. Retener una
feature solo si mejora Macro-F1 en el mismo split temporal y su disponibilidad
en inferencia está demostrada. No inferir el significado semántico de ninguna
clase hasta recibir el diccionario oficial.
