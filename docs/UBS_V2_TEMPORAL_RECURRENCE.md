# UBS V2: temporalidad y recurrencia

## Alcance y punto de partida

Trabajo exclusivo en `features/ginestar-v2-temporal`, desde
`0199a8b7c8b2c790a9d3447b156f2088708c2864` (también `origin/main` al comenzar,
24/09/2026). No se incorpora la auditoría independiente de `dev/carles` ni se
reescriben los componentes compartidos. Los únicos componentes nuevos son los
resúmenes temporales, su adaptador al heurístico existente y la comparación.
Se reutilizan `ubs.data`, `ClientFeatureBuilder`, `RecurrenceHeuristic`,
`ubs.evaluation` y el núcleo `evaluation.official`.

V1 se ejecutó **antes de modificar código** con seed 42 y `configs/ubs_v1.toml`;
solo se cambiaron las rutas de salida para conservar artefactos anteriores.
Train/valid/test oficiales: 2.000/1.000/1.000 clientes. Configuración: ventanas
7/14/30/60/90/180; TF-IDF 2.500; LR C=0,3/1/3, pesos none/balanced;
CatBoost 350 iteraciones, profundidad 6, learning rate 0,05. Reproducción:
**Macro-F1 0,2710242658492452**, accuracy 0,266, heurístico seleccionado;
**166,2291757 s**. Esta reproducción selecciona modelos y calibración contra
valid oficial: **no es un resultado independiente**. El refit train+valid y el
CSV de esa ejecución pertenecen a V1; V2 no usa test para seleccionar ni envía
ninguna submission.

## Auditoría de las features temporales V1

Notación: corte C; tiempos ordenados t1…tn; intervalos d; edad a=C−tn;
m=mediana(d); s=desviación poblacional(d). Un stream es la clave existente
`(client_id, description)`: no se presupone que sea un comercio limpio ni una
familia verdadera. Todas las duraciones siguientes se expresan en días UTC.

| Features existentes | Definición y datos | Poco historial / limitación |
|---|---|---|
| history_days, days_since_last_transaction | tn−t1, C−tn por cliente | Un evento produce span 0; no mide antigüedad de la cuenta |
| transactions_per_30d | 30n/max(tn−t1,1) | Un evento aparenta 30/mes; no cuenta inactividad hasta C |
| transactions/outgoing/frequency_last_wd | conteos en [C−w,C), frecuencia conteo/w | No ajusta exposición de historias cortas; transacciones de todas las familias |
| weekend_share, dow_j_share | fracción de eventos en fin de semana/día j | Un evento da concentración artificial; no verifica una cadencia |
| active_months, monthly_count_mean/std/cv | meses con eventos, media/desviación/CV de sus conteos | Omite meses sin actividad; std poblacional 0 con un mes |
| event_gap_mean/median/std | diferencias consecutivas de todos los eventos del cliente | Mezcla streams; std muestral; faltantes terminan en cero |
| median_interval_days, interval_std_days, interval_cv | mediana, std poblacional; s/max(m,1) por stream | Exige n≥2; un intervalo tiene std 0; timestamps repetidos pueden dar intervalo 0 |
| regularity, due_score | 1/(1+CV); exp(−abs(a−m)/max(m,7)) | Dos eventos aparentan regularidad perfecta; due_score no modela explícitamente el horizonte |
| base_recurrence_score | log(1+n) × regularity × due_score /(1+amount_cv) | Mantiene la estabilidad de importe de V1; no se cambia en esta línea temporal |
| regular_stream_count | número de streams con s≤3 | Cuenta todos los streams con solo dos eventos como regulares |
| periodicity_*_count | m en [4,10), [10,18), [18,45), [45,110), [110,400) | Bins amplios; el último se llama annual pero también incluye ~180 días |
| stream_*_mean/min, best/mean_recurrence_score | medias, mínimos, máximo de métricas de streams | Pierden identidad, soporte y dispersión entre streams |
| repeated_*_count, max/mean_description_appearances | cantidad de streams/eventos repetidos | Repetición no implica periodicidad |
| family_recency/frequency/regularity/recurrence_score | mínimo de edad, máximo de apariciones/regularidad, suma score×lift | Familia inferida por mapping supervisado; no es etiqueta por evento |

Los cálculos temporales de V1 usan historia anterior a C, con guardia de cutoff
en `transform` y en el loader. El mapping se aprende en `fit`: su reutilización
sobre los propios clientes de entrenamiento puede incluir su etiqueta. Esta
comparación evita ese riesgo evaluando siempre clientes ajenos a cada fit.
El runner V1 usa valid para tuning y selección; se documenta, sin modificarlo.
Tampoco se corrigen aquí sus limitaciones no temporales de importes/divisas.

## Evidencia previa a la implementación

Train contiene 147.459 transacciones y 50.720 streams: 17.669 con un evento,
9.325 con dos y 23.726 con tres o más. Cero timestamps repetidos por
cliente/stream, cero intervalos nulos y cero eventos en/después del cutoff.
El loader existente comprueba duplicados exactos, cobertura de IDs, clases y
disjunción de clientes entre las tres particiones.

V1 clasifica como regulares (std≤3) los **9.325** streams de dos eventos, frente
a solo **817** de los streams con ≥3. La ausencia de una segunda observación
de intervalo es el principal motivo para introducir soporte explícito.
Antes de implementar, los percentiles 10/50/90 de la mediana del intervalo
fueron 23,878/61,067/162,576 días; edad 13,340/84,049/255,383 días. Esto no
sostiene asumir periodicidad mensual universal.

Por target del cliente, las medianas de su mediana de intervalo están entre
60,35 y 62,80 días. `none` tiene menos eventos (64 frente a 74–79,5), pero no
carece de actividad ni de regularidad: su mediana del CV exploratorio
(std/media) es 0,406, inferior
a 0,461–0,487 de las otras clases. No se asigna el target del cliente a cada
transacción. Los agregados reproducibles siguientes excluyen los intervalos
sin soporte al medir dispersión; por ello su CV no coincide con el análisis
exploratorio inicial, que incluía ceros para los streams de dos eventos.

## Definición de V2 y decisiones congeladas

`temporal_streams` recibe únicamente cliente, descripción y timestamp. No
lee etiquetas, importes, MCC ni texto semántico. Ordena y deduplica tiempos
dentro del stream para estimar intervalos; conserva tanto transaction_count
como unique_event_count y duplicate_timestamp_count. Rechaza nulos y eventos
en/después del corte. Usa nanosegundos explícitos para no depender de la
resolución de datetime de pandas. El corte debe tener timezone.

| Bloque / feature | Definición |
|---|---|
| Recencia y volumen | n único, conteo original, k=n−1, C−tn, C−t1, tn−t1 |
| Intervalos | media, mediana, min/max(d); std poblacional, MAD=mediana(abs(d−m)), CV=std/media, MAD/m |
| Último intervalo | d último y abs(d último−m)/m |
| Soporte y confianza | u=k/(k+2); confianza=u/(1+CV) solo si k≥2, cero en otro caso |
| Periodicidad | por p en 7/14/28/30/31/90/180/365, exp(−mediana(abs(d−p))/(0,15p)), solo con k≥2 |
| Fase semanal/calendario | módulo de la media de exp(2πi·fase), fase weekday/7 o (día−1)/31, solo con n≥3 |
| Ventanas | conteos únicos en [C−w,C), w=30/60/90/180; ratio conteo/n |
| Tendencia | (n90/max(min(C−t1,90),1))/(n/max(C−t1,1)) |
| Próxima fecha | tn+m; si k≥2, 27≤m≤32 y concentración mensual≥0,95, siguiente mes calendario; preserva fin de mes si todos los eventos son fin de mes |
| Espera y atraso | espera=fecha−C; overdue=max(−espera,0) |
| Horizonte | indicador 0≤espera<h; evidencia=u·exp(−overdue/max(m,1)) si espera<h, cero en otro caso |

Con un evento no se inventan intervalos/fecha: NaN/NaT, indicador de fecha
conocida falso y evidencia cero. Con dos eventos se estima fecha, pero std,
MAD, CV y periodicidad quedan desconocidos. No se adelanta una fecha vencida
sumándole ciclos artificialmente. La evidencia de horizonte **no es una
probabilidad calibrada**. El extremo [C,C+h) es una convención explícita para
diagnósticos; el contrato no aclara su inclusión exacta. No sustituye targets
oficiales. Estacionalidad anual no es estimable con fiabilidad en un año.

El adaptador cambia **solo family_*_recurrence_score de las siete familias**.
Mantiene mapping, ocurrencias, lift, regularidad V1 y score de none. Para cada
stream repetido calcula factores f: intervals=0,5+confianza;
periodicity=0,5+u·max(proximidad); activity=0,5+clip(tendencia,0,2)/2;
horizon=0,5+evidencia. Para combinarlos usa su media geométrica, no su producto.
Por cliente/familia promedia f ponderando por el lift positivo existente y
multiplica el score V1 por ese promedio. Sin mapping aplica factor 1.
Bloques vacíos devuelven exactamente V1; nunca se rellena una familia
desconocida con una etiqueta de valid/test. La clase none conserva su significado
y su bias se calibra en clientes internos, no se usa como fallback de errores.

Estas fórmulas, rangos y seis variantes se fijaron antes de observar las
ablaciones. Las restantes estadísticas se conservan para diagnóstico, sin
añadir un clasificador ni hacer búsqueda de features contra valid.

## Protocolo experimental

1. Cinco folds estratificados de **clientes** de train, shuffle, seed 42.
   Cada fold: 1.600 clientes de desarrollo y 400 externos al fit.
2. Dentro de los 1.600: 1.280 para mapping y 320 para calibración, estratificados
   con seed 42+fold. Mapping y categorías se ajustan solo con esos 1.280.
   Se llama a `RecurrenceHeuristic.tune` sin modificar su grid: bias −2…2,5
   paso 0,25, temperaturas 0,5/0,75/1/1,5/2. Temperatura no altera argmax;
   se conserva para equivalencia con V1. No hay fitting de TF-IDF, escalado o
   imputación en las ablaciones del heurístico.
3. Congelada la calibración, se reajusta el mapping con los 1.600 y se mide en
   los mismos 400 para V1 y todos los bloques. Precalcular estadísticas puras
   por cliente no comunica información entre clientes ni aprende parámetros.
4. Macro-F1 siempre sobre las ocho clases, con el scorer compartido. Predicciones
   indexadas por client_id; aserciones de separación y cobertura OOF completa.
   Se guardan todos los folds, F1 por clase, distribuciones y matrices (filas
   reales, columnas predichas), media y desviación muestral (ddof=1).
5. Regla de promoción: ganar media y **cada uno de los cinco folds** frente a
   V1. En otro caso se conserva V1. Estos resultados se llaman selección
   interna; elegir una variante con ellos no produce una estimación independiente
   de la variante elegida. No se afirma significación estadística.
6. Después de congelar la decisión, se calibran dentro de train y diagnostican
   V1, combined y la seleccionada (sin duplicados) en valid oficial. Solo una
   pasada por versión de código comprobada, sin ajustar reglas a esos resultados.
   Valid ya estaba contaminada históricamente y por la reproducción V1.
7. Cortes 2025-07-01 y 2025-10-01 dentro de train, con 90 días completos:
   features solo antes del corte; target diagnóstico = primer evento observado
   del mismo stream en [corte,corte+90). No se reutiliza el target de enero.
   No hay ground truth de familia por evento ni definición suficiente para
   generar labels históricos oficiales; se reportan MAE de fecha, cobertura y
   conteos binarios, **no un Macro-F1 oficial ficticio**. El error de fecha está
   condicionado a que exista un evento futuro; se informa también de censura
   y nuevos streams para no ocultar el sesgo de cobertura.

Las particiones se registran mediante conteos y hashes, no IDs. El JSON ligero
incluye hashes de entradas/código, versiones, configuración y todas las métricas;
no contiene transacciones, modelos ni predicciones individuales.

## Reproducción

Desde la raíz del repositorio con el entorno instalado:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts/compare_ubs_temporal.py --config configs/ubs_v2_temporal.toml
```

El comando único ejecuta análisis, comparación V1/ablaciones, diagnóstico oficial
y cortes históricos. Escribe `outputs/metrics/ubs_v2_temporal/results.json`
(ignorado). No depende de los resultados guardados en Git. La reproducción
completa del runner original es `scripts/run_ubs_baseline.py --config
configs/ubs_v1.toml`; esa ejecución sí usa valid para selección y escribe el
CSV V1. No confundir ambos protocolos.

## Resultados medidos

Las tablas y conclusiones siguientes corresponden a la ejecución final del
código verificado. El JSON versionado conserva más decimales y todos los folds.

### Selección interna: todos los folds y ablaciones

| Variante | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Media | Std | Folds ganados |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | 0.442878 | 0.426100 | 0.485086 | 0.485194 | 0.470165 | 0.461885 | 0.026420 | 0 |
| intervals | 0.455199 | 0.447496 | 0.484349 | 0.505112 | 0.468635 | 0.472158 | 0.023145 | 3 |
| periodicity | 0.463105 | 0.445678 | 0.477284 | 0.510756 | 0.464810 | 0.472327 | 0.024255 | 3 |
| activity | 0.452971 | 0.435318 | 0.474668 | 0.503839 | 0.476697 | 0.468699 | 0.025962 | 4 |
| horizon | 0.449222 | 0.444294 | 0.466290 | 0.492755 | 0.483637 | 0.467240 | 0.021047 | 4 |
| combined | 0.463817 | 0.446628 | 0.477217 | 0.494337 | 0.466754 | 0.469750 | 0.017600 | 3 |

F1 por clase sobre las predicciones OOF concatenadas (no es la media de F1 de los folds):

| Variante | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | 0.475771 | 0.481400 | 0.471311 | 0.418719 | 0.307692 | 0.446115 | 0.435000 | 0.666052 |
| intervals | 0.480176 | 0.478936 | 0.490644 | 0.441805 | 0.346041 | 0.452261 | 0.435597 | 0.656280 |
| periodicity | 0.491071 | 0.484444 | 0.482618 | 0.436451 | 0.335329 | 0.447174 | 0.440191 | 0.663452 |
| activity | 0.478936 | 0.491150 | 0.469136 | 0.453012 | 0.312312 | 0.456576 | 0.432836 | 0.659735 |
| horizon | 0.461883 | 0.486239 | 0.491525 | 0.437055 | 0.338279 | 0.443325 | 0.432039 | 0.656163 |
| combined | 0.477679 | 0.478555 | 0.482328 | 0.450839 | 0.326284 | 0.457584 | 0.424390 | 0.664200 |

Distribución de predicciones OOF (2.000 clientes por variante):

| Variante | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | 264 | 267 | 274 | 215 | 114 | 204 | 175 | 487 |
| intervals | 264 | 261 | 267 | 230 | 143 | 203 | 202 | 430 |
| periodicity | 258 | 260 | 275 | 226 | 136 | 212 | 193 | 440 |
| activity | 261 | 262 | 272 | 224 | 135 | 208 | 177 | 461 |
| horizon | 256 | 246 | 258 | 230 | 139 | 202 | 187 | 482 |
| combined | 258 | 253 | 267 | 226 | 133 | 194 | 185 | 484 |

### Matriz OOF: v1

| Real / predicha | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 108 | 16 | 20 | 16 | 3 | 9 | 6 | 12 |
| gym | 14 | 110 | 11 | 14 | 5 | 13 | 12 | 11 |
| insurance | 24 | 17 | 115 | 11 | 5 | 13 | 13 | 16 |
| mobile | 20 | 20 | 15 | 85 | 9 | 22 | 6 | 14 |
| music | 22 | 27 | 26 | 14 | 48 | 15 | 12 | 34 |
| software | 15 | 16 | 24 | 17 | 9 | 89 | 7 | 18 |
| streaming | 16 | 27 | 28 | 19 | 11 | 16 | 87 | 21 |
| none | 45 | 34 | 35 | 39 | 24 | 27 | 32 | 361 |

### Matriz OOF: intervals

| Real / predicha | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 109 | 15 | 13 | 19 | 5 | 11 | 9 | 9 |
| gym | 14 | 108 | 10 | 13 | 5 | 16 | 14 | 10 |
| insurance | 21 | 17 | 118 | 13 | 10 | 10 | 13 | 12 |
| mobile | 18 | 18 | 15 | 93 | 11 | 17 | 10 | 9 |
| music | 22 | 29 | 24 | 13 | 59 | 11 | 16 | 24 |
| software | 18 | 13 | 22 | 18 | 9 | 90 | 10 | 15 |
| streaming | 17 | 24 | 26 | 22 | 13 | 16 | 93 | 14 |
| none | 45 | 37 | 39 | 39 | 31 | 32 | 37 | 337 |

### Matriz OOF: periodicity

| Real / predicha | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 110 | 13 | 15 | 20 | 3 | 10 | 9 | 10 |
| gym | 14 | 109 | 10 | 13 | 7 | 15 | 13 | 9 |
| insurance | 21 | 17 | 118 | 12 | 6 | 13 | 14 | 13 |
| mobile | 17 | 19 | 16 | 91 | 10 | 21 | 8 | 9 |
| music | 22 | 27 | 26 | 13 | 56 | 15 | 14 | 25 |
| software | 18 | 14 | 23 | 16 | 9 | 91 | 9 | 15 |
| streaming | 15 | 26 | 26 | 20 | 15 | 16 | 92 | 15 |
| none | 41 | 35 | 41 | 41 | 30 | 31 | 34 | 344 |

### Matriz OOF: activity

| Real / predicha | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 108 | 15 | 20 | 16 | 4 | 8 | 9 | 10 |
| gym | 14 | 111 | 10 | 14 | 5 | 15 | 11 | 10 |
| insurance | 23 | 16 | 114 | 8 | 14 | 12 | 12 | 15 |
| mobile | 18 | 19 | 17 | 94 | 10 | 19 | 5 | 9 |
| music | 18 | 28 | 25 | 15 | 52 | 15 | 13 | 32 |
| software | 16 | 16 | 21 | 16 | 10 | 92 | 7 | 17 |
| streaming | 18 | 27 | 26 | 19 | 12 | 17 | 87 | 19 |
| none | 46 | 30 | 39 | 42 | 28 | 30 | 33 | 349 |

### Matriz OOF: horizon

| Real / predicha | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 103 | 15 | 15 | 20 | 4 | 11 | 9 | 13 |
| gym | 14 | 106 | 10 | 13 | 7 | 15 | 14 | 11 |
| insurance | 19 | 13 | 116 | 13 | 10 | 13 | 14 | 16 |
| mobile | 18 | 19 | 14 | 92 | 10 | 17 | 7 | 14 |
| music | 22 | 26 | 22 | 15 | 57 | 13 | 12 | 31 |
| software | 17 | 14 | 22 | 16 | 11 | 88 | 8 | 19 |
| streaming | 17 | 25 | 23 | 18 | 14 | 15 | 89 | 24 |
| none | 46 | 28 | 36 | 43 | 26 | 30 | 34 | 354 |

### Matriz OOF: combined

| Real / predicha | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 107 | 15 | 15 | 18 | 5 | 8 | 8 | 14 |
| gym | 14 | 106 | 10 | 13 | 7 | 14 | 15 | 11 |
| insurance | 20 | 17 | 116 | 12 | 8 | 11 | 14 | 16 |
| mobile | 17 | 18 | 13 | 94 | 11 | 16 | 8 | 14 |
| music | 21 | 26 | 25 | 14 | 54 | 14 | 13 | 31 |
| software | 16 | 13 | 25 | 17 | 9 | 89 | 8 | 18 |
| streaming | 18 | 27 | 26 | 20 | 12 | 14 | 87 | 21 |
| none | 45 | 31 | 37 | 38 | 27 | 28 | 32 | 359 |

### Validation oficial: diagnóstico contaminado

| Protocolo / variante | Macro-F1 | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 original seleccionada en valid | 0.271024 | 0.288401 | 0.339921 | 0.323810 | 0.364217 | 0.238994 | 0.271357 | 0.247312 | 0.094183 |
| v1 | 0.140430 | 0.208696 | 0.085106 | 0.113821 | 0.157480 | 0.000000 | 0.081967 | 0.055046 | 0.421324 |
| combined | 0.153978 | 0.245902 | 0.081633 | 0.111111 | 0.176471 | 0.019802 | 0.097561 | 0.086957 | 0.412389 |

| Protocolo / variante | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 original seleccionada en valid | 230 | 132 | 111 | 209 | 66 | 95 | 89 | 68 |
| v1 | 26 | 20 | 24 | 23 | 7 | 18 | 12 | 870 |
| combined | 33 | 26 | 27 | 32 | 8 | 19 | 18 | 837 |

### Matriz valid: V1 original seleccionada en valid

| Real / predicha | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 46 | 5 | 5 | 15 | 4 | 3 | 7 | 4 |
| gym | 20 | 43 | 5 | 20 | 5 | 12 | 5 | 11 |
| insurance | 16 | 11 | 34 | 14 | 2 | 8 | 8 | 6 |
| mobile | 12 | 4 | 11 | 57 | 1 | 4 | 5 | 10 |
| music | 13 | 12 | 9 | 21 | 19 | 5 | 6 | 8 |
| software | 19 | 13 | 8 | 13 | 8 | 27 | 9 | 7 |
| streaming | 15 | 13 | 6 | 26 | 3 | 6 | 23 | 5 |
| none | 89 | 31 | 33 | 43 | 24 | 30 | 26 | 17 |

### Matriz valid: v1

| Real / predicha | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 12 | 2 | 0 | 0 | 1 | 0 | 1 | 73 |
| gym | 1 | 6 | 2 | 1 | 1 | 2 | 0 | 108 |
| insurance | 0 | 1 | 7 | 2 | 1 | 4 | 1 | 83 |
| mobile | 2 | 0 | 5 | 10 | 0 | 0 | 0 | 87 |
| music | 2 | 1 | 1 | 0 | 0 | 1 | 0 | 88 |
| software | 0 | 1 | 2 | 0 | 0 | 5 | 3 | 93 |
| streaming | 0 | 0 | 0 | 1 | 0 | 0 | 3 | 93 |
| none | 9 | 9 | 7 | 9 | 4 | 6 | 4 | 245 |

### Matriz valid: combined

| Real / predicha | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 15 | 2 | 0 | 2 | 0 | 0 | 1 | 69 |
| gym | 1 | 6 | 2 | 2 | 1 | 2 | 0 | 107 |
| insurance | 0 | 2 | 7 | 3 | 1 | 4 | 2 | 80 |
| mobile | 3 | 0 | 5 | 12 | 0 | 0 | 0 | 84 |
| music | 2 | 1 | 1 | 1 | 1 | 1 | 0 | 86 |
| software | 1 | 1 | 3 | 2 | 0 | 6 | 3 | 88 |
| streaming | 0 | 0 | 0 | 2 | 0 | 0 | 5 | 90 |
| none | 11 | 14 | 9 | 8 | 5 | 6 | 7 | 233 |

### Cortes históricos: fecha, cobertura y horizonte

| Corte | Streams | Con fecha estimable | Con evento futuro | Pares para error | Nuevos streams |
| --- | --- | --- | --- | --- | --- |
| 2025-07-01 | 33482 | 18368 | 16246 | 10317 | 9657 |
| 2025-10-01 | 43330 | 26419 | 19412 | 14290 | 7290 |

| Corte | MAE mediana V1 | MAE calendario V2 | Mediana AE V1 | Mediana AE V2 | Streams ajustados |
| --- | --- | --- | --- | --- | --- |
| 2025-07-01 | 63.659513 | 63.658705 | 50.525764 | 50.525764 | 94 |
| 2025-10-01 | 72.842133 | 72.839401 | 53.974708 | 53.983067 | 245 |

| Corte | TP | FP | FN | TN |
| --- | --- | --- | --- | --- |
| 2025-07-01 | 4726 | 3458 | 11520 | 13778 |
| 2025-10-01 | 6140 | 4730 | 13272 | 19188 |

### Estadísticas temporales V2 de train

| Estadística | P10 | P50 | P90 |
| --- | --- | --- | --- |
| unique_event_count | 1.000000 | 2.000000 | 6.000000 |
| number_of_intervals | 0.000000 | 1.000000 | 5.000000 |
| interval_mean | 29.552813 | 69.735203 | 161.866042 |
| interval_median | 23.877940 | 61.066863 | 162.575666 |
| interval_std | 11.506246 | 38.998601 | 85.095472 |
| interval_mad | 3.993484 | 22.206221 | 67.027101 |
| interval_cv | 0.199246 | 0.634595 | 0.996137 |
| days_since_last | 13.339640 | 84.049427 | 255.382595 |
| days_until_expected_next | -136.843345 | 3.637002 | 111.495822 |
| weekday_concentration | 0.178664 | 0.410123 | 0.748993 |
| monthday_concentration | 0.172955 | 0.454769 | 0.906454 |
| events_30d | 0.000000 | 0.000000 | 1.000000 |
| events_60d | 0.000000 | 0.000000 | 1.000000 |
| events_90d | 0.000000 | 1.000000 | 2.000000 |
| events_180d | 0.000000 | 1.000000 | 3.000000 |
| recent_to_history_rate | 0.000000 | 0.543557 | 1.388271 |

| Periodo (días) | Streams con ≥2 intervalos y proximidad ≥0,5 |
| --- | --- |
| 7 | 4 |
| 14 | 21 |
| 28 | 558 |
| 30 | 776 |
| 31 | 754 |
| 90 | 300 |
| 180 | 118 |
| 365 | 0 |

Streams con ajuste mensual: 342; fecha estimada en el horizonte: 12640.

| Target del cliente | total_events | streams | interval_median | interval_cv | interval_mad | days_since_last | events_90d |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 74.0000 | 26.0000 | 62.8013 | 0.6228 | 21.5149 | 87.5612 | 0.5000 |
| gym | 74.5000 | 26.0000 | 61.2711 | 0.6387 | 22.7655 | 87.1411 | 1.0000 |
| insurance | 79.5000 | 26.0000 | 60.7023 | 0.6407 | 21.8215 | 80.6876 | 1.0000 |
| mobile | 75.0000 | 25.0000 | 61.1971 | 0.6368 | 23.0956 | 84.5012 | 1.0000 |
| music | 77.0000 | 25.0000 | 60.3540 | 0.6463 | 21.2781 | 85.2420 | 1.0000 |
| software | 74.0000 | 26.0000 | 61.4234 | 0.6288 | 21.5181 | 85.1017 | 1.0000 |
| streaming | 74.0000 | 25.0000 | 61.0648 | 0.6436 | 24.1684 | 81.0753 | 1.0000 |
| none | 64.0000 | 24.0000 | 62.5268 | 0.6175 | 21.9539 | 86.2338 | 1.0000 |

Tiempo total de comparación final: **240.798360 s**; V1 original: **166.229176 s**. Versiones: numpy 2.3.5, pandas 2.3.3, scikit-learn 1.9.1, catboost 1.2.10.

## Conclusiones y riesgos pendientes

**No se demuestra una mejora consistente; no promover V2 para la entrega.**
Periodicity tiene la mejor media interna (0,472327 frente a 0,461885), pero
pierde en los folds 3 y 5. Combined alcanza 0,469750, también perdiendo en
esos dos folds. Los cuatro bloques y la combinación mejoran la media, pero
ninguno cumple la regla fijada de ganar todos los folds. No se reeligió una
seed ni se ajustaron umbrales después de observar estas tablas.

El diagnóstico oficial muestra un problema de transferencia: calibrada solo
en train, V1 predice 870/1.000 `none` (bias 1,0), combined 837 (bias 0,75).
El Macro-F1 cae a 0,140430 y 0,153978, respectivamente, pese al ~0,47 interno.
Esto es evidencia de fragilidad del protocolo/mapping/calibración al cambiar
partición, no una prueba causal de cuál de ellos falla. No se corrige mediante
tuning en valid. La V1 original de 0,271024 había elegido la calibración contra
esa misma valid: comparar sus 0,271 con el ~0,47 de train mezclaría protocolos.

Los ajustes de calendario son escasos y apenas cambian el MAE (menos de 0,003
días en ambos cortes). El error absoluto de ~64–73 días y la abundancia de
streams nuevos/ausentes indican que extrapolar simplemente el último intervalo
no describe bien el siguiente evento observado. La mediana del error incluso
empeora ligeramente en octubre; no se selecciona solo el corte favorable.
Los streams con pocos eventos y las descripciones opacas limitan la señal.

La recomendación operativa es **mantener el runner/configuración V1 existente**
mientras el equipo investiga el desajuste de distribución y la calibración.
La selección interna devuelve `v1` como control conservador; no significa
reemplazar la calibración del runner por el bias interno que aquí se diagnosticó.
La nueva rama aporta features, pruebas y evidencia negativa revisable, sin
alterar automáticamente el modelo de entrega. No hay métrica oficial independiente
ni promesa de ganancia en hidden test. Los cinco folds comparten train y mapping;
no son cinco datasets independientes. Elegir entre seis variantes también tiene
riesgo de selección múltiple. La falta de etiquetas por evento impide validar
históricamente las ocho familias oficiales: requiere información del organizador,
no targets derivados del mapping del propio modelo.

Se realizaron dos ejecuciones completas de comparación con las mismas reglas:
la segunda incorpora diagnósticos serializables y comprobaciones de casos sin
fecha futura. No se cambiaron bloques, parámetros, splits ni selección tras
la primera. Ambas dieron las mismas métricas de clasificación. Valid se consultó
en ambas, además de la reproducción V1; no se presenta como holdout nuevo.

Riesgo técnico pendiente, bajo: la segunda ejecución emitió una vez
`RuntimeWarning: overflow encountered in multiply` desde NumPy. Se repitieron
aisladamente los dos diagnósticos históricos y el resumen completo de 50.720
streams con `-W error::RuntimeWarning`: los tres terminaron correctamente, sin
reproducirlo. El JSON se serializó con `allow_nan=False`, y las métricas de
clasificación coinciden exactamente entre ejecuciones. No se suprime el aviso
ni se atribuye una causa no demostrada; queda pendiente localizar su origen
si reaparece en el entorno del equipo.

## Verificaciones y archivos

- Suite completa: **51 tests pasan**; incluye 14 casos nuevos temporales.
- Ruff global y `ruff format --check .`: correctos, 45 archivos formateados.
- Imports de módulos nuevos y evaluadores: correctos.
- CSV de reproducción V1 validado con `evaluation.official`: 1.000 clientes.
- `git diff --cached --check`: correcto. Pre-commit completo: Ruff, formato,
  trailing whitespace, YAML y tamaño de archivos pasan.
- `scripts/emergency_submission.py` conserva su SHA-256 inicial
  `AD525BAEF46EEE1B2E5B31AE2D5D774E680478D7EDE490B44B517B433C9E1E08`, sin staging.

Archivos propios: `configs/ubs_v2_temporal.toml`,
`scripts/compare_ubs_temporal.py`,
`src/transaction_forecasting/ubs/temporal_features.py`,
`src/transaction_forecasting/ubs/temporal_experiment.py`,
`tests/test_ubs_temporal.py`, este informe y
`docs/results/ubs_v2_temporal_results.json`.
Datasets, resultados individuales, CSV y cachés permanecen ignorados.
No se modifica ningún archivo V1 ni main; no se abre PR ni se envía submission.
