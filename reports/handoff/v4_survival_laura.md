# V4 survival — Laura

Rama: `exp/v4-survival-laura`. Base resuelta desde el checkout y README:
`origin/main` (promoción V3-A), SHA `051ce64a7cf0ab999f8aacb81fa405d5fa0257cf` (merge de promoción),
V3-A VALID Macro-F1 publicado `0.424111097737`, OOF `0.459793826869`.
Los valores `BASE_*` del encargo eran placeholders. No se modifica la baseline.

## Protocolo

- Streams del builder V3 `payment_streams`: pagos salientes de tarjeta por
  `(client_id, description, currency)`. Sin dependencia de Christian.
- Se eliminan `recency`, `due`, `score`, `recent` calculados por el builder al corte
  oficial; recencia se recalcula en cada cutoff. Features sólo con filas `< cutoff`.
- Seis cutoffs trimestrales: abril 2024 a julio 2025. Targets auxiliares observados
  en TRAIN: próxima aparición del mismo stream, evento en `[cutoff, cutoff+90d)`.
  El evento exactamente en día 90 queda fuera. Sólo se aceptan ventanas completas.
  Se presupone observación continua de TRAIN hasta 2026-01-01; no hay información
  independiente de cierre de cuenta. Nuevos streams sin historia no son candidatos.
- Cada cliente pesa uno: reparto uniforme entre sus snapshots y entre los streams
  de cada snapshot. Cinco folds estratificados por target de cliente, seed 42,
  iguales a la baseline. Ningún cliente comparte TRAIN/holdout, incluidos sus cutoffs.
- Modelo A: regresión logística regularizada de recurrencia; reparte su masa entre
  tres intervalos uniformemente al carecer de predicción temporal explícita.
- Modelo B: regresión logística multinomial de tiempo discreto con clases
  `[0,30)`, `[30,60)`, `[60,90)`, `>=90/sin evento`. Equivale a una distribución
  de supervivencia discreta; admite clases ausentes y conserva su orden.
- Temporal: count, recencia, gaps, MAD/std, último gap y ratios, ciclos observados
  y perdidos, estabilidad semanal/mensual, scores semanal/mensual/anual.
  Importe: mediana, CV, drift y ratio reciente. Contexto MCC/type/currency.
- Evidencia de familia: lift positivo suavizado existente, aprendido únicamente
  con clientes de entrenamiento del outer fold. Se usa al agregar, no como
  etiqueta real del stream ni como feature supervisada del modelo temporal.
  Confianza de identidad `1-exp(-sum(lift positivo))`: proxy no calibrado.
- Seasonal mirror es una ablación del modelo B: Q1 del año anterior, distancia a
  aniversario, mismo mes del año anterior, fase anual, número de eventos Q1.
  El score anual de gaps está también en el modelo sin mirror.

## Agregación y NONE

El modelo estima directamente P(evento en 90 días), incorporando inactividad.
Por ello el principal fija `P_active=1`, evitando contar dos veces la inactividad.
Se exportan ambas columnas y la evidencia de familia por stream.

Se comparan tres agregaciones TRAIN-only: competencia temporal por intervalos;
la misma con multiplicador de soporte `count/(count+2)` como proxy de actividad;
y masa total del horizonte sin prioridad temporal (`pooled`). NONE es el producto
de supervivencias de los streams ponderados por identidad. Dentro del intervalo
la asignación de familia es simétrica, proporcional a probabilidades condicionales;
no se fuerza un ganador por la fecha más temprana. Independencia entre streams y
reparto de empates son aproximaciones, no una calibración garantizada.

Sin evidencia de identidad o sin streams detectados, el modelo asigna NONE.
Es una limitación de cobertura: no demuestra ausencia real de eventos recurrentes.

Control median-gap: siguiente fecha mediana y decaimiento exponencial al acumular
retraso. No se interpreta como probabilidad calibrada. Híbridos predefinidos:
75% baseline + 25% cada alternativa temporal. Selección por Macro-F1 TRAIN OOF
entre todos los candidatos, incluida baseline; no hay nuevo binario NONE ni gates.
Los resultados de selección OOF son optimistas por comparación múltiple.

## Reproducción y artefactos

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/experiments/v4_survival_laura.py --phase oof
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/experiments/v4_survival_laura.py --phase valid
```

Directorio local ignorado: `outputs/metrics/v4_survival_laura/`.
`oof_<modelo>_probabilities.csv`: una fila por cliente TRAIN, orden oficial de
clases, para stacking. Los CSV por fold preservan membresía para stacking anidado.
`fold_*_streams.parquet`: diagnósticos al corte oficial.
`snapshots.parquet`: targets, features y pesos causales. No versionar estos archivos.
`oof_metrics.json`: métricas por cliente y fold y diagnósticos de recurrencia.
`frozen.json`: selección, receta, cutoffs y fingerprints antes de VALID.
`valid_started.json` se crea de forma exclusiva antes del acceso a VALID; impide
repetirlo incluso tras una interrupción. No se selecciona ni reajusta con VALID.

La caché OOF V3-A se verifica contra fingerprints de fuente y TRAIN y clientes
exactos por fold. Excepción comprobada: el cambio de promoción respecto a
`df9fe41` sólo alteró docstring y selector `predict`; se comparan los AST de `fit`
y `predict_components` antes de aceptar los CSV del brazo A.

## Interpretación de errores

Sin familias reales por transacción ni eventos oficiales post-cutoff no puede
separarse de forma fiable «familia correcta, horizonte incorrecto» de «stream
inactivo». `valid_errors.csv` distingue familia equivocada, discrepancia
NONE/evento y ausencia de stream, indicando esta falta de identificabilidad.
En pseudo-cutoffs sí existe próximo evento por stream: MAE se calcula sólo sobre
los eventos observados dentro de 90 días y nunca selecciona el modelo.
Top-event ranking accuracy mide si el stream de mayor P(evento) realmente recurre,
entre snapshots con alguna recurrencia; no pretende validar su familia semántica.
La precisión de alta confianza utiliza 0.8, fijado antes de evaluar, sólo como
métrica descriptiva. Coverage es fracción de predicciones positivas; también se
informa cobertura de alta confianza.

## Soporte efectivo del dataset

De los seis cortes predefinidos, tres no tienen historia y se omiten. Quedan:

| Corte | Filas | Clientes | Fracción de recurrencias |
|---|---:|---:|---:|
| 2025-01-01 | 6.233 | 1.872 | 0,3297 |
| 2025-04-01 | 16.064 | 1.995 | 0,3494 |
| 2025-07-01 | 26.225 | 2.000 | 0,3526 |

Total: 48.522 filas, 2.000 clientes, peso total exactamente uno por cliente.
`mirror_q1` y `same_month_last_year` valen cero en todos los pseudo-cutoffs.
Por tanto, la ablación no valida la utilidad de esas dos señales: falta soporte
histórico anual en entrenamiento. No se debe concluir que annual sea mejor ni peor
por su efecto en este dataset. Las otras features seasonal sí pueden variar.

## Resultados TRAIN OOF

| Modelo | Macro-F1 | Brier cliente | Logloss | NONE P/R |
|---|---:|---:|---:|---:|
| baseline | 0.459794 | 0.6709 | 1.4356 | 0.816/0.415 |
| median_gap | 0.374350 | 0.8221 | 2.3770 | 0.497/0.253 |
| recurrence_competing | 0.391724 | 0.7583 | 2.0125 | 0.624/0.114 |
| hazard_competing | 0.391541 | 0.7581 | 2.0068 | 0.636/0.114 |
| hazard_support | 0.423917 | 0.7071 | 1.5243 | 0.589/0.437 |
| hazard_seasonal_support | 0.418610 | 0.7068 | 1.5254 | 0.589/0.422 |
| hybrid_hazard_seasonal_support | 0.455289 | 0.6698 | 1.4231 | 0.746/0.444 |
| hybrid_median_gap | 0.457216 | 0.6823 | 1.4594 | 0.711/0.392 |

Se conserva **baseline V3-A**, ganadora OOF. El mejor híbrido global es el de
median-gap, sin mejora agregada. El híbrido hazard+seasonal+support mejora
solamente en 1 de cinco folds. No hay mejora consistente de Macro-F1.

Diagnóstico stream (media no ponderada de cinco folds; cada métrica interna
usa pesos de cliente salvo ranking):

| Modelo | AUROC | PR-AUC | Brier | Logloss | Top-event | MAE días |
|---|---:|---:|---:|---:|---:|---:|
| recurrence | 0.6140 | 0.4411 | 0.2133 | 0.6166 | 0.5275 | 22.8365 |
| hazard | 0.6140 | 0.4412 | 0.2133 | 0.6166 | 0.5283 | 22.2704 |
| hazard_seasonal | 0.6137 | 0.4415 | 0.2133 | 0.6166 | 0.5279 | 22.2708 |

La señal temporal es modesta (AUROC ~0,614); el mirror no mejora este diagnóstico.
La precisión de familias y la calibración de NONE dependen también del mapping
de identidad y de la independencia aproximada de streams.

Antes de VALID se recalculó logloss sobre los mismos CSV OOF para corregir
el orden lexicográfico implícito de sklearn: se codificaron las clases con índices
numéricos en orden oficial. No cambió ningún modelo, predicción, Macro-F1 ni
selección. `frozen.json` conserva las huellas de ejecución y las huellas finales
del runner, que también incorporó caché de snapshots; corrección registrada.

## VALID: una evaluación tras freeze

| Modelo | Macro-F1 | Evento accuracy | Coverage | NONE P/R |
|---|---:|---:|---:|---:|
| baseline | 0.424111 | 0.773 | 0.714 | 0.615/0.601 |
| median_gap | 0.253210 | 0.621 | 0.624 | 0.386/0.495 |
| recurrence_competing | 0.323805 | 0.709 | 0.968 | 0.531/0.058 |
| hazard_competing | 0.324291 | 0.709 | 0.968 | 0.531/0.058 |
| hazard_support | 0.311002 | 0.663 | 0.630 | 0.441/0.556 |
| hazard_seasonal_competing | 0.331371 | 0.710 | 0.971 | 0.552/0.055 |
| hybrid_hazard_seasonal_support | 0.418457 | 0.773 | 0.700 | 0.610/0.625 |
| hybrid_median_gap | 0.389100 | 0.729 | 0.654 | 0.532/0.628 |

Baseline reproducida exactamente: **0,4241110977365**. Se mantiene V3-A.
El mejor híbrido observado en VALID (no seleccionado) es hazard+seasonal+support,
0,418457; el mejor temporal solo es hazard+seasonal+competing, 0,331371.
La ablación support que mejor funcionaba sola en OOF transfiere peor en VALID.

| Familia | F1 baseline | F1 hazard support | F1 híbrido hazard seasonal support |
|---|---:|---:|---:|
| cloud | 0.482 | 0.409 | 0.476 |
| gym | 0.442 | 0.257 | 0.416 |
| insurance | 0.427 | 0.386 | 0.427 |
| mobile | 0.468 | 0.333 | 0.462 |
| music | 0.324 | 0.186 | 0.311 |
| software | 0.335 | 0.220 | 0.325 |
| streaming | 0.307 | 0.204 | 0.313 |
| none | 0.608 | 0.492 | 0.617 |

| Modelo | Brier cliente | Logloss | Coverage ≥0,8 | Precisión ≥0,8 |
|---|---:|---:|---:|---:|
| baseline | 0.7008 | 1.5300 | 0.000 | N/A |
| hazard_support | 0.7648 | 1.7145 | 0.004 | 0.500 |
| hazard_seasonal_competing | 0.8255 | 2.2731 | 0.001 | 0.000 |
| hybrid_hazard_seasonal_support | 0.7077 | 1.5481 | 0.000 | N/A |

Diagnóstico de decisiones (no ground truth de estado latente del stream):

| Modelo | Sin stream | Positivos con familia equivocada | NONE falso | Evento falso |
|---|---:|---:|---:|---:|
| baseline | 0 | 312 | 110 | 117 |
| hazard_support | 0 | 303 | 207 | 130 |
| hybrid_hazard_seasonal_support | 0 | 313 | 117 | 110 |

En hazard_support, 104 NONE falsos conservan la familia real como mejor clase
positiva: compatibles con error de horizonte/actividad o escala de evidencia.
Los eventos falsos pueden deberse a streams inactivos, identidad incorrecta o
horizonte errado; sin eventos futuros por stream no se pueden desambiguar.

## Handoff y recomendación

No promover el hazard ni los híbridos a baseline. Se entrega una señal temporal
OOF reproducible, con causalidad y masa NONE explícitas, pero no una mejora
competitiva validada. La poca historia anual, la identidad débil y el cambio
de distribución entre pseudo-cutoffs y corte oficial limitan la transferencia.
Cualquier integración posterior con merchant intelligence debe abrir otro
experimento TRAIN-only; no reutilizar esta VALID para ajustar pesos o thresholds.

Verificación final: **96 tests**, `ruff check .`, `ruff format --check .`,
`pre-commit run --all-files` e imports correctos. Ocho tests específicos cubren
causalidad, horizonte, pesos, aislamiento, determinismo y orden de clases/logloss.
Datos, modelos y probabilidades permanecen ignorados; sólo se versionan código,
tests y este handoff. No se ha rehecho el none gate anterior.
