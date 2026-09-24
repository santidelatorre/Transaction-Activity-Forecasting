# UBS V2: descripciones y comercios aproximados

Experimento de Javi, 24 de septiembre de 2026. Código aislado en
`ubs/text_v2.py` y `scripts/run_ubs_text_v2.py`; el runner y la configuración
de V1 no cambian. Se usaron los 2.000 clientes oficiales de train y los 1.000
de validación, con historias anteriores a `2026-01-01`. El target es
`target_next_recurring_merchant` y la métrica principal es Macro-F1 de las
ocho clases fijas. La columna `merchant` no existe: se utiliza la descripción
normalizada como aproximación de la identidad del comercio.

## Hipótesis y experimentos

La normalización reduce variaciones de mayúsculas, acentos, fechas, referencias
y números, conservando palabras del comercio. Se compararon TF-IDF de palabras
(unigramas y unigramas/bigramas), TF-IDF de caracteres 3–5 y atributos de
concentración de descripciones. Las estadísticas de comercios incluyen número
de descripciones distintas, entropía, cuota de la descripción dominante,
proporción recurrente, frecuencia en train y asociaciones por clase. Estas
últimas se calculan por clientes y, para las filas de train, excluyen la
contribución del propio cliente.

Todos los modelos de la tabla usan los 218 agregados V1, salvo `v1_selected`,
que es la predicción elegida por el runner V1. La regresión logística usa
`C=0.3`, `class_weight=balanced`, semilla 42, `tol=1e-5` y máximo 2.000 iteraciones.
TF-IDF usa `min_df=2`, `max_df=0.98`, máximo 2.500 términos de palabras o
3.500 de caracteres. Los vocabularios, imputaciones y escalados se ajustan
sólo con train. La variante combinada usa la mejor de dos representaciones
de texto elegida en validación.

| Variante | Macro-F1 | Δ frente a V1 | Accuracy | Features | Vocabulario | Entreno s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 seleccionada: heurística | 0.271024 | 0 | 0.266 | 32 | — | — |
| Agregados V1, control ML | 0.199439 | -0.071585 | 0.273 | 218 | 0 | 0.69 |
| Componente V1, TF-IDF word 1–2 | 0.200045 | -0.070979 | 0.280 | 2.538 | 2.320 | 1.77 |
| Normalizado, unigramas | 0.197970 | -0.073054 | 0.281 | 288 | 70 | 1.19 |
| Normalizado, word 1–2 | 0.200045 | -0.070979 | 0.280 | 2.538 | 2.320 | 1.86 |
| Normalizado, char 3–5 | 0.196875 | -0.074149 | 0.281 | 1.123 | 905 | 2.85 |
| Agregados V1 + descripción como comercio | 0.169151 | -0.101873 | 0.285 | 245 | 0 | 0.80 |
| Word 1–2 + comercio aproximado | 0.169339 | -0.101685 | 0.287 | 2.565 | 2.320 | 1.97 |

La normalización no alteró la representación word 1–2 en estos datos: el
lector V1 ya convierte las descripciones a minúsculas y no se observaron
referencias variables que cambiasen el vocabulario. Los n-grams de caracteres
tampoco aportaron mejora. Las cifras de entrenamiento excluyen construcción
de features y lectura del dataset; son tiempos aproximados del clasificador.

## Clases y cobertura

| Clase | Soporte valid | F1 V1 | F1 word 1–2 | Δ word | F1 word + comercio | Δ combinado |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 89 | 0.2884 | 0.1852 | -0.1032 | 0.0952 | -0.1932 |
| gym | 121 | 0.3399 | 0.2051 | -0.1348 | 0.1216 | -0.2183 |
| insurance | 99 | 0.3238 | 0.1515 | -0.1723 | 0.0840 | -0.2398 |
| mobile | 104 | 0.3642 | 0.1818 | -0.1824 | 0.1611 | -0.2031 |
| music | 93 | 0.2390 | 0.1224 | -0.1165 | 0.1438 | -0.0952 |
| software | 104 | 0.2714 | 0.2166 | -0.0548 | 0.1765 | -0.0949 |
| streaming | 97 | 0.2473 | 0.1486 | -0.0987 | 0.1447 | -0.1026 |
| none | 293 | 0.0942 | 0.3891 | +0.2949 | 0.4277 | +0.3336 |

Todos los clientes tienen algún texto, pero disponer de texto no garantiza
que identifique la siguiente familia recurrente. El clasificador textual
mejora `none` y empeora las siete familias; la ganancia de accuracy refleja
en parte ese cambio hacia la clase mayoritaria. Las matrices de confusión
de todas las variantes y F1 por clase completos se guardan en
`outputs/metrics/ubs_text_v2/` (fuera de Git).

## Términos y diversidad

Entre las descripciones de train aparecen asociaciones claras: `cloud access`
para cloud, `urban gym`/`fitness monthly` para gym, `safe cover` para insurance,
`phone contract` para mobile, `audio streaming` para music,
`productivity suite` para software y `video access` para streaming. El
informe generado muestra el lift y número de clientes de los ocho términos
principales de cada clase. Se cuentan dentro de cada descripción, para no
crear bigramas artificiales entre transacciones adyacentes. Son asociaciones
con el target del cliente, no etiquetas de cada movimiento; pueden reflejar
palabras literales del generador sintético.

La mediana de descripciones normalizadas distintas por cliente pasa de
aproximadamente 24–26 en train a 30–36 en validación. La entropía mediana
también sube de alrededor de 3.0 a 3.2–3.4. Esta diferencia de distribución
limita el valor de frecuencias y asociaciones exactas aprendidas en train.
La alta cardinalidad de descripciones introduce estimaciones frágiles;
agregar etiquetas de validación o movimientos posteriores al corte causaría
leakage.

## Recomendación y reproducción

No integrar estas variantes como sustituto de V1: ninguna supera su
Macro-F1. Mantener la heurística y estudiar después si señales textuales
locales al *stream* recurrente, en lugar de todo el documento del cliente,
ayudan a separar la siguiente familia de `none`. Validar esa hipótesis con
otro corte o test ciego antes de adoptar un cambio.

```powershell
.venv\Scripts\python.exe scripts/run_ubs_baseline.py --config configs/ubs_v1.toml
.venv\Scripts\python.exe scripts/run_ubs_text_v2.py --config configs/ubs_text_v2.toml
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m pre_commit run --all-files
```

La V1 se ejecuta primero para guardar su predicción de validación como
referencia exacta. Si no está disponible, el runner V2 usa el componente
logístico V1 como referencia temporal e indica ese cambio en `results.json`.
El experimento V2 no genera un CSV de entrega.
