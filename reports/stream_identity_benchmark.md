# Auditoría y benchmark de Stream Identity frente a V1/V2

Fecha: 2026-09-25. Rama: `research/import-v2-stream-identity`.
Modelo importado auditado: `e4aa4c58175242e198cefd32d6ac4558145523af`;
su implementación predictiva se había congelado en `ffc656e`.

## Alcance y criterio de aceptación

La reproducción numérica y la validez de una evaluación sin selección sobre
VALID son dos cuestiones diferentes. Este informe aplica expresamente la regla
solicitada: un resultado que influyó en la selección usando etiquetas VALID no
sirve como confirmación independiente, aunque el cálculo sea reproducible.

**Dictamen estricto: INVALID.** El desarrollo importado consultó VALID y rechazó
los sesgos de decisión por su resultado en ese conjunto. Esto contamina la
selección histórica; no demuestra fuga temporal ni inclusión de etiquetas VALID
en el ajuste supervisado. El número observado se presenta abajo como comparación
descriptiva sobre el conjunto común, sin prometer rendimiento de TEST.

## Inspección y procedencia de las baselines

Se inventariaron los 78 archivos versionados, los módulos de `src/ubs_recurrence`,
scripts de investigación y producción, todos los tests actuales, informes,
datos locales, métricas y predicciones existentes, y las referencias históricas
de Git. No existe `AGENTS.md` aplicable en el repositorio ni sus directorios
antecesores. Los directorios de cachés preexistentes se preservaron.

Esta rama contiene el historial importado, no los runners de las baselines
anteriores. Inicialmente no contenía `configs/`, `run_ubs_baseline.py`,
`run_ubs_v2.py`, `run_ubs_v3.py`, `validate_submission.py` ni `test_ubs_v1.py`.
`data/sample_submission.csv` tampoco es la ubicación efectiva: el contrato
histórico está en `data/raw/ubs_2026/sample_submission.csv`.

Se inspeccionaron los objetos Git sin cambiar de rama:

- V1 original: `0199a8b7c8b2c790a9d3447b156f2088708c2864`.
- V2 congelada: `96409b7940a991fbda5235b85ffeb40652a4087b`.
- Reauditoría y comparación histórica: `172356a22c3471077a8fa7a09060960091ea4cd7`.
- Corrección posterior de aislamiento de selección de V1: `50640f2`.
- Protocolo V3 y decisión final V4: objetos de `integration/v4-synthesis`,
  leídos únicamente. V4 conserva V3-A, Macro-F1 histórico 0.4241110977,
  y entrega entrenada solo con TRAIN. Ese resultado no se reentrenó aquí.

**Limitación de V1/V2 que no debe ocultarse:** el V1 que produjo 0.271024
llamaba a `RecurrenceHeuristic().tune(x_valid, y_valid)` y elegía modelo/mezcla
sobre VALID. No corresponde al protocolo posterior corregido de `50640f2`.
V2 fija su receta en el runner, pero los informes históricos reconocen selección
previa mediante experimentos sobre VALID. El informe histórico
`reports/v1_vs_v2_comparison.md` en `172356a` lo declara para ambos modelos.
Por tanto no existe una interpretación honesta de esos números históricos como
holdout prístino. No se alteró V1 ni V2 para favorecer al modelo importado.

En este benchmark se reconstruye la **receta ganadora ya congelada** de V1,
sin repetir su búsqueda sobre VALID: mapeo de descripciones ajustado con TRAIN,
heurística con `none_bias=-1.0`, `temperature=1.0`.
V2 se ajusta de nuevo con su clase histórica intacta: 75% CatBoost de historia
(300 árboles, profundidad 4, tasa 0.05, clases balanceadas, semilla 42) y
25% heurística de periodicidad. Las features y modelos V1 de `0199a8b` y
`96409b7` son idénticos. La extracción de objetos se guarda en un directorio
ignorado; no se hace checkout, worktree, merge ni modificación de referencias.

## Datos, clientes, split y target

Se utilizan los siete archivos del dataset oficial, verificados byte a byte
contra `reports/data_manifest.json`. Sus seis archivos comunes tienen los
mismos SHA-256 que la procedencia del benchmark V2 local. El archivo adicional
de preentrenamiento sin etiquetas se utiliza para los perfiles de precios.

| Split | Clientes | Transacciones | Extremos de ID (no contiguos) | Timestamp m?nimo UTC | Timestamp m?ximo UTC |
| --- | --- | --- | --- | --- | --- |
| train | 2000 | 147459 | C000001 ? C003999 | 2024-11-07 00:07:06+00:00 | 2025-12-31 23:57:50+00:00 |
| valid | 1000 | 73898 | C000000 ? C003998 | 2024-11-07 00:09:28+00:00 | 2025-12-31 23:55:23+00:00 |
| test | 1000 | 75761 | C000004 ? C003989 | 2024-11-07 00:33:43+00:00 | 2025-12-31 23:58:30+00:00 |
| unlabeled_pretrain | 10000 | 749935 | U000000 ? U009999 | 2024-11-07 00:06:22+00:00 | 2025-12-31 23:57:56+00:00 |

La pertenencia es exacta, no una nueva partición aleatoria:

- TRAIN: los IDs de `train_labels.csv`, iguales a los de `train_transactions.jsonl`.
- VALID: los IDs de `valid_labels.csv`, iguales a los de `valid_transactions.jsonl`.
- TEST: los IDs de `sample_submission.csv`, iguales a los de `test_transactions.jsonl`.
- Pretrain: los IDs de `unlabeled_pretrain_transactions.jsonl`, disjuntos de los otros tres.

Los listados completos ordenados se generan como `*_ids.txt` dentro de la
ejecución; sus hashes están en el JSON de métricas. Los intervalos de ID de la
tabla no implican continuidad. Los hashes de los archivos oficiales fijan la
pertenencia completa de manera reproducible.

Cutoff UTC: **2026-01-01 00:00:00**, exclusivo para features.
Target oficial: `target_next_recurring_merchant`, siguiente familia recurrente
en los 90 días siguientes, con las ocho clases exactas
`cloud, gym, insurance, mobile, music, software, streaming, none`.
Se usan las etiquetas entregadas; no se reconstruye un target proxy ni se
infiere a partir de TEST. El generador y sus reglas de desempate no están publicados
en los artefactos del repositorio.

La evaluación final siempre ajusta con los 2.000 TRAIN y puntúa los 1.000 VALID.
No se estratifica de nuevo el split oficial. El desarrollo importado usó cinco
folds estratificados por cliente dentro de TRAIN; las tres vistas aumentadas
del cliente permanecen juntas en el lado de entrenamiento del fold.

## Arquitectura, features y preprocesado

La clase evaluada es `FamilyForecaster`. Su componente llamado `legacy` o
`v1` en el repositorio importado pertenece a su propia investigación; **no es
la baseline UBS V1 de 0.271024**.

1. JSONL crudo, MCC como texto, timestamps UTC y orden estable por cliente/tiempo.
   Normalización textual a minúsculas, eliminación de caracteres no alfabéticos,
   espacios y sustituciones fijas de abreviaturas. No hay embeddings ni
   vectorizador supervisado nuevo en esta receta.
2. Solo pagos de tarjeta salientes para descubrir streams; exclusión textual
   fija de gastos de fondo como alimentación, viajes y comercio general.
   Grupos por cliente/moneda e importes conectados en log-importe, tolerancia
   0.035 y al menos tres eventos. Grupos amplios por familia/MCC admiten dos eventos.
3. Reglas semánticas y plantillas fijas aportan evidencia de familia. Se incluyen
   importes, dispersión/cambio, recencia, intervalos, regularidad, fase de calendario,
   extrapolación lineal del siguiente evento, recuentos de ventanas 30/60/90/180 días,
   diversidad de descripciones y coincidencias semánticas/MCC. Las extrapolaciones
   futuras se calculan desde historia pasada; no leen eventos futuros.
4. Perfiles de precio por familia: cuantiles 0.005/0.995 y mediana de anclas
   semánticas/MCC en los 10.000 historiales unlabeled. La asignación suave combina
   semántica, MCC y plausibilidad de precio. No utiliza targets de ningún split.
5. Evidencia contextual por cliente y comparaciones entre candidatos; actividad
   y periodicidades aproximadas 14/28/30/60/90 días; matching de devoluciones
   por moneda e importe, recencia de reembolso, comisiones y patrones horarios.
   Un reembolso es evidencia predictiva, no prueba de cancelación.
6. Ocho filas candidatas por cliente, incluidas `none`; 221 features en el bloque
   compacto final. `family_index` es el índice de la familia candidata de cada
   fila, no la etiqueta verdadera del cliente. `client_id` solo agrupa, ordena
   y alinea; no se pasa como columna predictiva. Los índices sí fijan el orden
   determinista de asignación de ruido, sin mapa supervisado de identidad.
7. Missing features se representan con `-999`. El detector de `none` agrega
   min/max/media sobre candidatos, retirando `family_index` e `is_none`: 657
   features para el detector binario.

Aumento congelado con semilla 2026: vista original, `valid_like` y `test_like`.
Sus tasas respectivas de masking/MCC/fondo/variantes textuales son
(0.30, 0.18, 0.06, 0.15) y (0.50, 0.30, 0.10, 0.25). Estas hipótesis fueron
informadas por auditorías **sin etiquetas** de la distribución de VALID/TEST;
se declara ese uso transductivo de covariables, sin considerarlo prueba de
generalización a una distribución nunca observada.

Ensemble congelado:

- 75%: promedio de semillas 42, 17 y 2026; cada una tiene un ranker LightGBM
  LambdaRank y clasificador binario de `none`. Cada estimador usa 500 árboles,
  15 hojas, tasa 0.035, `min_child_samples=120`, L2=10, fracción de columnas 0.95,
  cuatro hilos. Se conservan todas las semillas, sin elegir la favorable.
- 25%: promedio de dos rankers LightGBM (asignación dura/suave, 450 árboles,
  15 hojas, tasa 0.035, mínimo 105, L2=5, columnas 0.9, semilla 42) y un
  XGBoost con asignación suave (500 árboles, profundidad 4, tasa 0.04,
  `min_child_weight=10`, filas 0.85, columnas 0.9, L2=10, semilla 42,
  `rank:pairwise`, hist, CUDA, cuatro pares por muestra).
- Softmax por cliente, sustitución jerárquica de masa `none`, media de scores
  normalizados y argmax. No hay threshold ni sesgos ajustados en esta auditoría.
  Son seis expertos/nueve estimadores; los scores no garantizan calibración.

## Auditoría de leakage y del 0.61

El valor original exacto es **0.6194934236214421**, accuracy **0.647**, en
`reports/final_metrics.json`. Es distinto de 0.613157 de OOF con corrupción
`test_like` y de 0.7698 de una tarea histórica auxiliar. Ninguno de estos
dos últimos se usa como score del challenge.

| Riesgo investigado | Hallazgo |
| --- | --- |
| Score sobre TRAIN | La nueva evaluación predice clientes disjuntos del ajuste y lo exige en código. |
| Información posterior al cutoff | Los cuatro archivos de transacciones pasan la comprobación estricta; ningún evento posterior. |
| Target en features | Las funciones predictivas no aceptan labels; test de envenenamiento de columna target en todos los bloques finales. |
| Leakage de IDs | IDs excluidos de features; comprobación de renombrado y reordenación. |
| Clientes/historias duplicados | Sin solapamiento entre cuatro splits, sin duplicados exactos de eventos ni historias completas. |
| Split o clientes diferentes | Mismos archivos hash de V2 y cobertura exacta de todos los VALID. |
| Target/cutoff distintos | Etiquetas originales y cutoff original; no target derivado ni proxy. |
| Subconjunto favorable | No se eliminan clientes ni clases; soportes y distribución completos abajo. |
| Macro-F1 erróneo/clases ausentes | Evaluador histórico de ocho clases, división indefinida cero; contrastado con sklearn en el evaluador importado. |
| Etiquetas reutilizadas al predecir | Las dos réplicas persisten VALID y TEST antes de abrir VALID labels para scoring. |
| Semilla favorable | Mismas tres semillas del modelo congelado, todas incluidas; no búsqueda nueva. |
| Selección histórica sobre VALID | **Sí**: rechazo de sesgos de decisión tras consultar el primer holdout; cambios posteriores al primer lote. |
| Etiquetas TEST | No disponibles ni utilizadas; el sample contiene placeholders. |
| Reproducibilidad | Dos reconstrucciones independientes desde crudo; comprobación exacta de probabilidades y clases para VALID y TEST. |

Evidencia concreta de selección: `reports/final_protocol.md`,
`reports/final_candidate_config.json`, `reports/research_decisions.md` y
`reports/holdout_access_log.jsonl`. El primer lote comparó control, ensemble
y calibrated (0.177187, 0.587318 y 0.586551). La decisión final abandonó los
sesgos porque no confirmaron su mejora en ese holdout. Un segundo lote evaluó
el predictor final; una tercera lectura fue reproducción. Congelar antes de
la segunda lectura no deshace lo aprendido de la primera.

**INVALID se refiere al criterio de confirmación sin selección sobre VALID.**
No significa que la aritmética del score sea falsa ni que se haya probado
inclusión de futuro o etiquetas VALID en las features. Tampoco se puede medir
cuánto del incremento procede de esa selección sin un nuevo holdout intacto.
La comparación homogénea de clientes/target/cutoff/métrica sigue siendo útil,
con esa limitación expresamente visible y también aplicable a V1/V2 históricas.

## Resultados homogéneos e incertidumbre

| Modelo | Macro-F1 | Accuracy | ? vs V2 |
| --- | --- | --- | --- |
| V1 | 0.271024266 | 0.266 | -0.120525190 |
| V2 | 0.391549456 | 0.424 | +0.000000000 |
| Stream Identity | 0.619493424 | 0.647 | +0.227943968 |

Delta Stream Identity vs V1: **+0.348469158**.

| Estimaci?n | IC bootstrap 95% |
| --- | --- |
| V1 | [0.242851, 0.297697] |
| V2 | [0.359100, 0.420607] |
| Stream Identity | [0.584452, 0.649504] |
| delta_stream_vs_V2 | [0.191362, 0.264364] |

El valor original se reproduce exactamente: **0.6194934236214421 / 0.647**.
Las nuevas predicciones V1 y V2 coinciden con sus 1.000 predicciones hist?ricas,
verificadas despu?s del nuevo ajuste. Las dos r?plicas Stream Identity coinciden
exactamente en todas las probabilidades y clases de VALID y TEST. El CSV de
probabilidades VALID tiene SHA-256
`757b5fc2d4e71e585f232dae28f3a509078083ba40b8d2d49338ed97b376613e`.
La ejecuci?n completa tard? 971.7 segundos, incluyendo auditor?a, dos reconstrucciones,
baselines, bootstrap y validaci?n de entrega; no es una medida aislada de inferencia.

Bootstrap pareado de clientes, 2.000 réplicas, semilla 20260925, percentiles
2.5/97.5%, siempre sobre las ocho clases. Los mismos clientes remuestreados se
usan en los tres modelos. Condiciona en modelos ya entrenados y no captura
selección histórica, incertidumbre de entrenamiento ni shift de TEST.

### V1 ? m?tricas por clase

| Clase | Precision | Recall | F1 | Support real | Predicciones |
| --- | --- | --- | --- | --- | --- |
| cloud | 0.200000 | 0.516854 | 0.288401 | 89 | 230 |
| gym | 0.325758 | 0.355372 | 0.339921 | 121 | 132 |
| insurance | 0.306306 | 0.343434 | 0.323810 | 99 | 111 |
| mobile | 0.272727 | 0.548077 | 0.364217 | 104 | 209 |
| music | 0.287879 | 0.204301 | 0.238994 | 93 | 66 |
| software | 0.284211 | 0.259615 | 0.271357 | 104 | 95 |
| streaming | 0.258427 | 0.237113 | 0.247312 | 97 | 89 |
| none | 0.250000 | 0.058020 | 0.094183 | 293 | 68 |

### V2 ? m?tricas por clase

| Clase | Precision | Recall | F1 | Support real | Predicciones |
| --- | --- | --- | --- | --- | --- |
| cloud | 0.507042 | 0.404494 | 0.450000 | 89 | 71 |
| gym | 0.435065 | 0.553719 | 0.487273 | 121 | 154 |
| insurance | 0.358108 | 0.535354 | 0.429150 | 99 | 148 |
| mobile | 0.395683 | 0.528846 | 0.452675 | 104 | 139 |
| music | 0.216667 | 0.139785 | 0.169935 | 93 | 60 |
| software | 0.323944 | 0.442308 | 0.373984 | 104 | 142 |
| streaming | 0.317460 | 0.206186 | 0.250000 | 97 | 63 |
| none | 0.600897 | 0.457338 | 0.519380 | 293 | 223 |

### Stream Identity ? m?tricas por clase

| Clase | Precision | Recall | F1 | Support real | Predicciones |
| --- | --- | --- | --- | --- | --- |
| cloud | 0.635417 | 0.685393 | 0.659459 | 89 | 96 |
| gym | 0.613445 | 0.603306 | 0.608333 | 121 | 119 |
| insurance | 0.651163 | 0.565657 | 0.605405 | 99 | 86 |
| mobile | 0.670103 | 0.625000 | 0.646766 | 104 | 97 |
| music | 0.593023 | 0.548387 | 0.569832 | 93 | 86 |
| software | 0.563107 | 0.557692 | 0.560386 | 104 | 103 |
| streaming | 0.593023 | 0.525773 | 0.557377 | 97 | 86 |
| none | 0.709480 | 0.791809 | 0.748387 | 293 | 327 |

### Distribuci?n real y de predicciones en VALID

| Clase | Real | Real % | V1 | V2 | Stream Identity |
| --- | --- | --- | --- | --- | --- |
| cloud | 89 | 8.9% | 230 | 71 | 96 |
| gym | 121 | 12.1% | 132 | 154 | 119 |
| insurance | 99 | 9.9% | 111 | 148 | 86 |
| mobile | 104 | 10.4% | 209 | 139 | 97 |
| music | 93 | 9.3% | 66 | 60 | 86 |
| software | 104 | 10.4% | 95 | 142 | 103 |
| streaming | 97 | 9.7% | 89 | 63 | 86 |
| none | 293 | 29.3% | 68 | 223 | 327 |

### Matriz de confusi?n ? V1

Filas: verdad. Columnas: predicci?n.

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

### Matriz de confusi?n ? V2

Filas: verdad. Columnas: predicci?n.

| Real / predicha | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 36 | 5 | 11 | 6 | 4 | 10 | 5 | 12 |
| gym | 7 | 67 | 9 | 9 | 5 | 7 | 5 | 12 |
| insurance | 4 | 6 | 53 | 4 | 2 | 7 | 7 | 16 |
| mobile | 0 | 7 | 15 | 55 | 3 | 8 | 0 | 16 |
| music | 3 | 14 | 16 | 12 | 13 | 12 | 11 | 12 |
| software | 5 | 12 | 10 | 12 | 2 | 46 | 4 | 13 |
| streaming | 2 | 17 | 10 | 10 | 17 | 13 | 20 | 8 |
| none | 14 | 26 | 24 | 31 | 14 | 39 | 11 | 134 |

### Matriz de confusi?n ? Stream Identity

Filas: verdad. Columnas: predicci?n.

| Real / predicha | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 61 | 3 | 1 | 3 | 2 | 5 | 7 | 7 |
| gym | 5 | 73 | 6 | 4 | 2 | 6 | 5 | 20 |
| insurance | 4 | 8 | 56 | 5 | 5 | 5 | 5 | 11 |
| mobile | 1 | 5 | 6 | 65 | 5 | 5 | 2 | 15 |
| music | 4 | 6 | 7 | 5 | 51 | 8 | 2 | 10 |
| software | 6 | 7 | 1 | 4 | 5 | 58 | 5 | 18 |
| streaming | 6 | 7 | 0 | 4 | 8 | 7 | 51 | 14 |
| none | 9 | 10 | 9 | 7 | 8 | 9 | 9 | 232 |

## Reproducción, cambios y entrega

Entorno: Python 3.13.7 y versiones fijadas en `requirements-lock.txt`;
Ruff 0.16.8. XGBoost usa NVIDIA RTX 3060 Laptop, driver 610.78.
CPU puede producir diferencias respecto de CUDA; no se seleccionó dispositivo
por su resultado. La procedencia exacta, los hashes de código evaluado y de
datos están en `outputs/metrics/stream_identity/metrics.json`.
El HEAD registrado durante la ejecución es el commit importado indicado arriba,
con las adaptaciones de esta entrega todavía sin commit; no se presenta como
un checkout limpio. Los hashes identifican los archivos realmente ejecutados.

Comando completo ejecutado desde la raíz, tras instalar las dependencias y
el paquete (`python -m pip install -e ".[dev]"`):

```powershell
python -X utf8 scripts/run_ubs_stream_identity.py --run-name benchmark_train_only_20260925 --device cuda --replicas 2
```

No usa cachés de features, modelos binarios ni predicciones previas para el
modelo importado. Reconstruye perfiles de precios y todas las features y ajusta
los nueve estimadores en cada réplica. También reajusta las recetas históricas
V1/V2. Las predicciones históricas conservadas sirven únicamente para contrastar
que las nuevas coinciden cliente a cliente; no sustituyen el entrenamiento.
Las predicciones anteriores, datos y submissions siguen intactos.

La ejecución se conserva en
`outputs/metrics/stream_identity/benchmark_train_only_20260925/`.
Comando de verificación final de hashes y recálculo de las tres métricas:

```powershell
python -X utf8 scripts/run_ubs_stream_identity.py --run-name benchmark_train_only_20260925 --verify
python scripts/validate_submission.py --submission outputs/predictions/submission_stream_identity.csv --sample data/raw/ubs_2026/sample_submission.csv
```

Los directorios de ejecución son inmutables: volver a entrenar requiere un
directorio de ejecución nuevo y destinos libres para los artefactos de entrega.
En un clon sin datos, `python scripts/prepare_data.py` recupera el archivo
oficial fijado; usar después `--data-dir data/raw`. Los objetos históricos de
V1/V2 deben estar presentes en Git (clon completo del repositorio).

Se reutilizan loaders y features importados con un parámetro de ruta explícito,
evitando duplicar o sobrescribir los datos históricos. Se recuperó el evaluador
oficial interno de V1/V2 como `src/ubs_recurrence/official.py`, con AST idéntico,
y su CLI validator con la única adaptación de imports. El validator importado
original admitía reordenación; el histórico recuperado exige el orden del sample.
El predictor conserva la misma lógica/hiperparámetros: los módulos de streams,
ranking, compact, augmentation, templates, payment context y model mantienen
equivalencia AST tras normalizar imports, literales dict y el nombre de una
variable no usada. La cantidad de cambios de formato se explica por la
obligación de hacer pasar Ruff en toda la rama importada.

Para entrega se sigue el protocolo vigente de esta rama (`final_protocol.md`)
y de V4: **ajuste solo con los 2.000 TRAIN**. Las dos réplicas son refits completos
de esa receta. No se adopta el antiguo refit TRAIN+VALID de V1/V2, ni se calcula
una métrica VALID después de incorporar esas etiquetas. Las predicciones TEST
también se producen antes de leer labels VALID. Esto permite generar una
submission técnicamente válida aunque el score histórico no cumpla el criterio
de holdout sin selección.

CSV: `outputs/predictions/submission_stream_identity.csv`.

- Columnas exactas: `client_id,predicted_next_recurring_merchant`.
- 1.000 filas y 1.000 IDs ?nicos; exactamente los del sample y en su orden.
- Cero IDs faltantes o sobrantes, duplicados, nulls, blancos o clases ilegales.
- Ajuste con 2.000 TRAIN; etiquetas VALID excluidas del fit.
- Validator hist?rico recuperado: **PASS**.
- SHA-256: `e226823a1cacb1579d3f7f40a2636e3f60f8126e362e4495d5a0e9cdb5ebadf7`.
- Coincide con la submission importada, sin sobrescribirla.

| Clase | Predicciones TEST |
| --- | --- |
| cloud | 96 |
| gym | 105 |
| insurance | 97 |
| mobile | 88 |
| music | 64 |
| software | 97 |
| streaming | 79 |
| none | 374 |

No se subió el CSV al challenge. No hay score de TEST ni se hace una afirmación
sobre el leaderboard. Se versionan únicamente código, tests, informe, referencia
compacta de predicciones históricas y los pequeños CSV/JSON de entrega; datos,
snapshots, cachés y registros de ejecución permanecen ignorados.

## Tests y verificación final

El primer `pytest` sin rutas falló con **118 errores de colección** porque
recogía tests de snapshots históricos bajo `outputs/`. Los 16 tests originales
de esta rama sí pasaban. Se fijó `testpaths=["tests"]` y se ejecutaron los tests
históricos UBS explícitamente desde su código intacto: no se ocultaron fallos
de tests actuales. Ruff inicialmente encontraba 277 incidencias incluyendo
cachés ajenas (87 en código/tests de esta rama). Se excluyeron únicamente
directorios de artefactos/cachés y se corrigió el código actual, incluido formato,
imports no usados y el binding explícito de una variable de bucle.

| Comprobaci?n final | Resultado |
| --- | --- |
| `pytest -q` | 21 passed, exit 0 |
| Tests UBS hist?ricos del commit V2 | 48 passed, exit 0 |
| `python -m ruff check .` | PASS, exit 0 |
| `python -m ruff format --check .` | PASS, exit 0 |
| Benchmark `--verify` | PASS: hashes y rec?lculo de las tres m?tricas, exit 0 |
| CLI `validate_submission.py` | PASS, exit 0 |
| Inspecci?n manual independiente del CSV | Esquema, 1.000 IDs/filas, orden, unicidad, no nulls/blancos, ocho clases: PASS |
| Probabilidades de las r?plicas | Finitas, no negativas, normalizadas, argmax consistente; igualdad exacta VALID/TEST |
| Datos comunes con V2 | Seis SHA-256 id?nticos |
| Artefactos previos protegidos | 16 archivos comprobados por SHA-256; cero cambios |
| Hook pre-commit local | Ruff check, format check y tests: PASS |

Los 48 tests hist?ricos se ejecutaron desde el snapshot intacto, con su `src`
en `PYTHONPATH`, mediante:

```powershell
python -m pytest --import-mode=importlib tests/test_ubs_v1.py tests/test_ubs_v2.py tests/test_official.py tests/test_submission_validator.py tests/test_evaluation_compatibility.py tests/test_ubs_temporal.py tests/test_ubs_text_v2.py -q
```

Los logs se conservan en `outputs/stream_identity_audit/`. Durante la inferencia,
XGBoost avis? de entrada CPU para un booster CUDA y us? su fallback DMatrix.
El wrapper PowerShell con redirecci?n report? exit 1 tras registrar ese aviso
como `NativeCommandError`; el runner alcanz? `Complete`, no hubo traceback
Python y se escribi? el recibo de todas las etapas. La verificaci?n posterior
independiente termin? con exit 0 y comprob? todos los hashes y scores. El aviso
no se silenci? ni se cambi? el dispositivo para buscar otro resultado.


La reproducción completa incluye dos ajustes desde cero. Una ejecución inicial
se interrumpió antes de abrir VALID labels al resolver la diferencia histórica
de procedimiento de submission; sus archivos se preservaron como incompletos.
No se eligió una receta por su score en esa ejecución.

## Conclusión y límites

El modelo reconstruido supera descriptivamente V1 y V2 sobre exactamente el
mismo VALID: Macro-F1 **0.619493424**, accuracy **0.647**, delta sobre V2
**+0.227943968**. El intervalo pareado de ese delta queda por encima de cero.
El supuesto ~0.61 se reproduce **num?ricamente**; la etiqueta formal solicitada
es **INVALID**, porque el desarrollo hist?rico us? VALID para decidir entre
postprocesados. No se encontr? fuga temporal, target como feature, intersecci?n
de clientes, entrenamiento con VALID ni error de Macro-F1. La submission es
v?lida y regenerable, sin convertir esa validez de formato en un score de TEST.

El resultado de este benchmark no puede convertir un VALID ya consultado en
un nuevo holdout independiente. Para certificar el incremento fuera de esta
selección harían falta etiquetas externas nunca utilizadas y un protocolo
congelado antes de verlas. Los bootstrap son descriptivos; el dataset es
sintético; el generador del target es incompleto; TEST muestra shift adicional;
no se validan fechas/importes futuros ni impacto bancario real.

STREAM_IDENTITY_061_STATUS: INVALID

STREAM_IDENTITY_MACRO_F1: 0.6194934236214421

STREAM_IDENTITY_ACCURACY: 0.647

DELTA_VS_V2: 0.22794396771088793

SUBMISSION_VALID: YES
