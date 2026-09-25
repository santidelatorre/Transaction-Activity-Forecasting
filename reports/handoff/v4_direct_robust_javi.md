# V4 direct robust: handoff de Javi

## Estado y protocolo congelado

- Rama: `exp/v4-direct-robust-javi`; SHA inicial `051ce64a7cf0ab999f8aacb81fa405d5fa0257cf`; árbol inicial limpio.
- Los parámetros `BASE_BRANCH`, `BASE_SHA`, `BASE_MACRO_F1` y `BASE_RUNNER` llegaron como `__PLACEHOLDER__`. La baseline promovida documentada en `README.md` es V3-A: TRAIN OOF Macro-F1 `0.459793826869`; VALID `0.424111097737`. El control V2 histórico en VALID es `0.391549455911`.
- Cinco folds estratificados por cliente, semilla 42, idénticos al runner V3. Ningún vocabulario, frecuencia de cliente, IDF o fuente de reemplazos genéricos se ajusta fuera del fold de entrenamiento.
- Selección predefinida: `0.25 * clean F1 + 0.35 * medium F1 + 0.40 * severe F1 - 0.10 * media de SD de F1 entre folds`.
- Corrupción local reproducible, ajustada con historiales TRAIN del fold: dropout de filas de description, enmascaramiento de streams completos, fragmentación de alias y sustitución por descripciones frecuentes entre clientes. `medium` y `severe` son escenarios simulados, no el shift oficial. Módulo aislado en `LocalCorruptor` para sustituirlo por la suite común.
- R0: presencia binaria de description completa; R1: conteo truncado a cinco; R2: R1 más TF-IDF de caracteres; R3: R2 más description × MCC; R4: R3 más estadísticas genéricas y específicas; R5: R2 más esas estadísticas, sin interacciones MCC. La comparación R4/R5 y la selección final se hacen sólo con OOF TRAIN.
- Modelos: ComplementNB y LogisticRegression multinomial de scikit-learn. Cada uno tiene entrenamiento clean en R0–R5; en R4/R5 se comparan clean, clean + dos vistas corruptas y la misma mezcla con peso total uno por cliente. No se aplican gates ni thresholds.

## Resultados

La configuración congelada por TRAIN OOF fue **ComplementNB / R4 / normalized**. Su robust_score fue `0.47662`. En la tabla, el entrenamiento `augmented` da peso total tres por cliente; `normalized` da peso total uno. Las veinte combinaciones medidas, incluidas las negativas, están en `oof_results.json`.

| Candidato TRAIN OOF | Clean | Medium | Severe | Robust score |
| --- | ---: | ---: | ---: | ---: |
| ComplementNB / R4 / normalized | 0.5074 | 0.4781 | 0.4614 | 0.4766 |
| ComplementNB / R4 / augmented | 0.4987 | 0.4667 | 0.4494 | 0.4657 |
| ComplementNB / R4 / clean | 0.5064 | 0.4555 | 0.3792 | 0.4351 |
| ComplementNB / R3 / clean | 0.5063 | 0.4572 | 0.3920 | 0.4409 |
| ComplementNB / R1 / clean | 0.4975 | 0.4549 | 0.4023 | 0.4422 |
| LogisticRegression / R4 / normalized | 0.4557 | 0.4258 | 0.3409 | 0.3968 |
| LogisticRegression / R4 / clean | 0.4449 | 0.3406 | 0.2002 | 0.3082 |

La normalización de vistas elevó el F1 severe de R4/NB de `0.3792` a `0.4614`, con clean prácticamente igual (`0.5064` frente a `0.5074`). R2/R5 clean y LogisticRegression en general degradaron mucho bajo corrupción. No se promovieron por su clean OOF.

Los cinco F1 por fold de la selección fueron:

| Vista | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | .4789 | .4997 | .5410 | .5280 | .4811 |
| Medium | .4511 | .4662 | .5020 | .5062 | .4600 |
| Severe | .4305 | .4551 | .4744 | .4664 | .4742 |

La media de las tres desviaciones estándar entre folds fue `0.0213`.

## VALID oficial y comparación

Se guardaron `valid_probabilities.csv` y `valid_predictions.csv` **antes** de leer `valid_labels.csv`. Se evaluó una vez; no hubo reintento ni ajuste posterior.

| Modelo | TRAIN OOF clean Macro-F1 | VALID Macro-F1 |
| --- | ---: | ---: |
| Baseline actual V3-A | 0.4598 | **0.4241** |
| Direct robust congelado | **0.5074** | **0.0764** |
| V2 histórico | — | 0.3915 |

La caída frente a V3-A en VALID fue `-0.3478`. El clasificador colapsó hacia `none` (989/1000 predicciones). Es un resultado **negativo** para integración o stacking inmediato: el buen OOF y la robustez bajo esta corrupción local no predijeron la transferencia a VALID. No se observaron errores complementarios útiles en las sólo once predicciones positivas de VALID.

| Clase | OOF F1 | VALID F1 | Predicciones VALID | V3-A VALID F1 |
| --- | ---: | ---: | ---: | ---: |
| cloud | .472 | .044 | 2 | .482 |
| gym | .510 | .032 | 3 | .442 |
| insurance | .514 | .020 | 2 | .427 |
| mobile | .443 | .000 | 0 | .468 |
| music | .425 | .000 | 0 | .324 |
| software | .493 | .019 | 2 | .335 |
| streaming | .452 | .040 | 2 | .307 |
| none | .750 | .456 | 989 | .608 |

Matriz de confusión VALID; filas reales y columnas predichas en orden `cloud, gym, insurance, mobile, music, software, streaming, none`:

```text
cloud       2 0 0 0 0 0 0  87
gym         0 2 0 0 0 0 0 119
insurance   0 1 1 0 0 0 0  97
mobile      0 0 0 0 0 0 0 104
music       0 0 0 0 0 1 0  92
software    0 0 0 0 0 1 0 103
streaming   0 0 0 0 0 0 2  95
none        0 0 1 0 0 0 0 292
```

Confianza media / entropía media / accuracy: OOF clean `0.810 / 0.545 / 0.548`; VALID `0.968 / 0.109 / 0.300`. La confianza extrema con baja accuracy confirma una mala calibración fuera de TRAIN.

## Cobertura, coste y asociaciones TRAIN

Cobertura de descriptions completas por transacción: holdout OOF `99.47–99.53%`, VALID `99.17%`. En VALID, sólo `71.58%` de las **descriptions únicas** aparece en TRAIN. La mediana de descriptions únicas por cliente es 25 en TRAIN y 33 en VALID, aunque las transacciones medias por cliente son 73.73 y 73.90. Esto sugiere un shift en la cola de nombres; no demuestra que sea la única causa del colapso.

Dimensión R4: `9,337–9,420` columnas en folds; `9,919` en el ajuste completo. El OOF tardó aproximadamente 15 minutos de CPU local; el ajuste y evaluación VALID, `155.4 s`. Pico observado del proceso durante OOF: cerca de `0.95 GB`; `tracemalloc` de VALID marcó `350.6 MiB` de asignaciones Python, sin toda la memoria nativa.

Las mayores contribuciones del modelo ajustado en TRAIN están en `valid_results.json`. Ejemplos **sólo de TRAIN**: `member plan plus` aparece en 2 clientes, ambos etiquetados gym; `dining service` aparece en 2, ambos cloud. Son señales específicas pero extremadamente frágiles. `digital service` aparece en 419 clientes: 297 none y 122 repartidos entre las otras clases, por lo que es un nombre genérico y ambiguo. En los doce pesos más altos por clase predominan nombres completos e interacciones description × MCC; ninguna de las cinco estadísticas genéricas figura en ese top. La clasificación `kind` del JSON es superficial para interacciones MCC: el handoff usa frecuencia de clientes TRAIN para identificar nombres genéricos. Ningún peso implica causalidad ni justifica una regla manual.

## Artefactos y verificación

`outputs/metrics/v4_direct_robust_javi/` contiene `summary.json`, `oof_results.json` con todos los candidatos, `frozen_selection.json`, probabilidades OOF clean/medium/severe alineadas por `client_id` y columnas en orden oficial, y las probabilidades, predicciones y métricas VALID. Son archivos locales ignorados por Git; los OOF clean (`355,744` bytes) quedan disponibles para stacking posterior, pero este candidato no debe incluirse sin una explicación del fallo en VALID.

Pasaron seis pruebas focalizadas para fit-only, aislamiento de cliente, folds deterministas, pesos de vistas, ruta clean y orden de clases. Ruff check y format check de los tres archivos nuevos pasaron.

## Limitaciones previstas

- La selección sobre OOF puede optimizar el mismo conjunto usado para reportarla; VALID es la estimación independiente.
- La corrupción local no mide directamente el shift real y no debe interpretarse como resultado del reto.
- No se añadió pérdida de consistency: replicar labels entre vistas y normalizar peso por cliente era la comparación sencilla predefinida.
- Las features de coeficiente alto son asociaciones estadísticas, no causas ni reglas manuales.
- El entorno original `.venv` apunta a un Python 3.12 ausente. La reproducción local usa Python 3.14 con dependencias instaladas en `.runtime314/` (ignorado por Git), fuera del rango `<3.14` declarado por el proyecto.
