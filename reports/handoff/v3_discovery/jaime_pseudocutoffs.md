# Jaime — supervisión temporal con pseudo-cutoffs

## Resultado

**PSEUDO-CUTOFFS HELP MODERATELY**

Se generaron **7,338 pseudo-cutoffs** y **72,741 filas candidato** para **1,991 clientes**. El ranker congelado por OOF fue `hist_gradient_boosting`.

## Configuración y distribución temporal

Cutoffs por cliente: media 3.69, mediana 4.0, rango 1–4. Fechas: 2025-03-08T00:00:00+00:00 a 2025-10-03T00:00:00+00:00.

Se exigen 120 días y 12 eventos de historia, separación de 60 días, dos apariciones previas por stream y 90 días futuros completamente observables.

## Métricas OOF agrupadas por cliente

| modelo | top-1 | top-2 | recall none | F1 ocurre 90d | MAE próxima ocurrencia (días) |
| --- | ---: | ---: | ---: | ---: | ---: |
| next_date_rule | 0.1494 | 0.2899 | 0.0141 | 0.9854 | 33.82 |
| logistic_pointwise | 0.2067 | 0.3530 | 0.0141 | 0.9855 | 21.68 |
| hist_gradient_boosting | 0.2142 | 0.3532 | 0.0986 | 0.9851 | 21.65 |

### Distribución de pseudo-cutoffs por mes

| mes | pseudo-cutoffs |
| --- | ---: |
| 2025-03 | 1,497 |
| 2025-04 | 84 |
| 2025-05 | 1,763 |
| 2025-06 | 119 |
| 2025-07 | 1,844 |
| 2025-08 | 122 |
| 2025-09 | 1,885 |
| 2025-10 | 24 |

El modelo logístico y el boosting son formulaciones pointwise de ranking: cada conjunto contiene los streams recurrentes y una fila `none`, y se ordena por la probabilidad de ser el primer stream que reaparece.

### Top-1 del modelo seleccionado por número de candidatos

| candidatos | pseudo-cutoffs | top-1 |
| --- | ---: | ---: |
| 0 | 3 | 1.0000 |
| 1 | 34 | 0.5294 |
| 2 | 174 | 0.4080 |
| 3-4 | 990 | 0.3434 |
| 5+ | 6,137 | 0.1858 |

## Aplicación al cutoff oficial y posible efecto en Macro-F1

La selección del ranker usa únicamente OOF de TRAIN. Después se ajusta en todos los pseudo-ejemplos de TRAIN y se aplica a VALID en 2026-01-01. La familia se obtiene mediante el mapping fijo `ClientFeatureBuilder.description_lift_`, aprendido solo con TRAIN; no se optimiza la clasificación de familia.

| estrategia | Macro-F1 VALID | accuracy VALID |
| --- | ---: | ---: |
| v2_temporal_heuristic | 0.2726 | 0.2650 |
| next_date_rule | 0.1603 | 0.2930 |
| pseudo_cutoff_ranker | 0.1055 | 0.2790 |

Cambio del ranking aprendido frente a la regla simple: **-0.0548 Macro-F1**. Este diagnóstico no es un holdout independiente: VALID ya fue reutilizado en V2, aunque no interviene en la selección del ranker de este experimento.

## Análisis de leakage

- Features: solo transacciones con `timestamp < T`.
- Labels: solo primeras ocurrencias en `[T, T+90d)` y únicamente después de construir features.
- Observabilidad: todo cutoff termina como máximo en 2025-10-03, dejando el horizonte completo antes de 2026-01-01.
- Split: cinco folds por `client_id`; todos los cutoffs y candidatos de un cliente permanecen juntos.
- Aplicación oficial: ranker y mapping se ajustan en TRAIN; VALID solo se usa para el diagnóstico final.
- Tests: invariancia de features ante cambios futuros, límites del horizonte y aislamiento de clientes.

## Coste computacional

Tiempo total: **319.3 s**; generación: 196.9 s; OOF: 32.7 s. El experimento usa modelos tabulares pequeños y no usa deep learning.

## Reproducibilidad

Commit base: `5884ddd07a19b2e19b44b4fd55afdf4eb6dcf749`. Semilla: `42`. Dataset TRAIN SHA-256: `ba0902b33b1181921ee5b1f8f445693201c0968ca52d70a92e12cbb114e7d70a`.
