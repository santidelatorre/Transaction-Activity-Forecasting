# V4 soft candidate scorer — Santiago

**Decisión: NO PROMOVER sobre V3-A.** El scorer real aprende una mejora estable
en TRAIN OOF y bajo corrupción, pero no la transfiere a VALID. Se entrega la
implementación, la evaluación congelada y sus artefactos; la baseline queda intacta.

## Resultados

| Modelo | TRAIN OOF Macro-F1 | Accuracy | Top-2 recall | Corrupción Macro-F1 | VALID Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V3-A congelada | 0.459794 | 0.4710 | 0.7300 | 0.454978 | 0.424111 |
| Raw family | 0.254798 | 0.2500 | 0.4615 | 0.268526 | 0.192719 |
| Logística | 0.499181 | 0.5225 | 0.7620 | 0.483360 | no evaluada |
| **CatBoost seleccionado** | **0.542662** | **0.5630** | **0.8050** | **0.519199** | **0.400662** |

Son 2.000 clientes TRAIN y 1.000 VALID. Rank accuracy = accuracy, por definición.
La selección de CatBoost se congeló con TRAIN, antes de abrir labels VALID.
No se evaluó logística en VALID después de observar el resultado de CatBoost.
Las probabilidades de control coinciden exactamente con los cinco artefactos
OOF de V3-A existentes (máxima diferencia absoluta = 0 en cada fold).

| Fold externo | V3-A | Logística | CatBoost |
| --- | ---: | ---: | ---: |
| 1 | 0.422837 | 0.441151 | 0.497450 |
| 2 | 0.449722 | 0.514847 | 0.576839 |
| 3 | 0.476690 | 0.506719 | 0.539670 |
| 4 | 0.474125 | 0.521800 | 0.547994 |
| 5 | 0.465166 | 0.503924 | 0.547751 |

La mejora OOF agregada sobre V3-A es +0.082868. Bootstrap pareado descriptivo:
IC percentil 95% [+0.062780, +0.102493], 2.000 remuestreos, seed 42; no corrige
selección de modelo ni incorpora incertidumbre de reajuste. La caída por corrupción
del ranker es -0.023463; aun así supera a V3-A corrupta en +0.064221.

### F1 por clase

| Clase | OOF V3-A | OOF ranker | VALID V3-A | VALID ranker |
| --- | ---: | ---: | ---: | ---: |
| cloud | 0.463964 | 0.515222 | 0.481928 | 0.400000 |
| gym | 0.469526 | 0.518337 | 0.442478 | 0.413146 |
| insurance | 0.474438 | 0.604255 | 0.427273 | 0.474747 |
| mobile | 0.450704 | 0.513447 | 0.468085 | 0.393258 |
| music | 0.395062 | 0.480198 | 0.323810 | 0.270833 |
| software | 0.429224 | 0.527964 | 0.334802 | 0.330097 |
| streaming | 0.444934 | 0.498896 | 0.306569 | 0.317241 |
| none | 0.550499 | 0.682977 | 0.607945 | 0.605974 |

### Dónde gana en TRAIN

| Segmento OOF | Clientes | V3-A Macro-F1 | Ranker Macro-F1 |
| --- | ---: | ---: | ---: |
| Sin familia con evidencia | 16 | 0.112069 | 0.116667 |
| Una familia | 109 | 0.588033 | 0.669544 |
| Varias familias | 1.875 | 0.450440 | 0.533891 |
| Especificidad textual alta | 1.971 | 0.459019 | 0.542172 |
| Especificidad textual baja | 29 | 0.102041 | 0.110577 |
| Confianza recurrente alta | 614 | 0.421193 | 0.515015 |
| Confianza recurrente baja | 1.386 | 0.472659 | 0.553750 |

Entre los 1.338 positivos con varias familias, la accuracy pasa de 0.489537 a
0.557549. El ranker corrige 289 errores de V3-A y pierde 105 aciertos; ambos fallan
en 769 clientes y discrepan en 547. Los segmentos sin evidencia o con poca
especificidad son pequeños: no justifican conclusiones firmes.

La disponibilidad de candidatos es siempre 100%; la cobertura de **evidencia hard
para el target positivo** es 95.0107% OOF y 93.2107% VALID. No deben confundirse.
Una ausencia de evidencia no impide predecir esa clase.

### Por qué no se promueve

VALID cae -0.023449 Macro-F1 frente a V3-A. La accuracy baja de 0.461 a 0.455,
aunque top-2 sube de 0.685 a 0.699. Solo mejoran insurance y streaming en F1.
Entre 988 clientes con varias familias, Macro-F1 baja de 0.425003 a 0.401749;
entre los 702 positivos ambiguos, accuracy baja de 0.404558 a 0.343305.
El ranker corrige 103 errores de V3-A, pierde 109 aciertos y comparte 436 errores;
hay complementariedad, pero no se ha probado ni seleccionado una mezcla.

`none` conserva casi el mismo F1 VALID (0.605974 vs 0.607945), con un cambio
importante de error: los falsos positivos de `none` suben de 110 a 197 y sus falsos
negativos bajan de 117 a 80. Aquí FP significa **predecir none con target positivo**;
FN significa predecir una familia con target none. En OOF, los FP habían bajado
56→49 y los FN 349→262. En VALID hay 410 predicciones none frente a 286 de V3-A,
sin que se haya ajustado ninguna prevalencia.

El diagnóstico posterior al freeze encuentra cambios de evidencia:

| Feature / grupo | TRAIN OOF | VALID |
| --- | ---: | ---: |
| Familias con evidencia, media | 3.8375 | 4.9260 |
| Máxima recurrencia, media | 0.8929 | 0.7662 |
| Suma de recurrencia, media | 3.8590 | 2.9676 |
| Baseline P(none), media | 0.1373 | 0.2085 |
| Clientes con alta confianza recurrente | 30.7% | 13.8% |
| Eventos históricos, media | 73.7295 | 73.8980 |

Estos cambios son compatibles con un problema de transferencia de las features
y de la decisión none, más que con falta de candidatos; **no demuestran la causa**.
Parte de la diferencia puede proceder del tamaño del fit del mapping. No se
reajusta el modelo después de leer VALID ni se presenta la mejora OOF como mejora
desplegable confirmada. Se supera 0.50 en OOF, no en VALID; el criterio ideal del
encargo no se cumple. Mantener V3-A para integración.

## Base y alcance

- Rama de trabajo: `exp/v4-soft-candidate-santiago`.
- `BASE_BRANCH = origin/baseline/v4-frozen` (referencia local disponible).
- `BASE_SHA = 051ce64a7cf0ab999f8aacb81fa405d5fa0257cf`.
- `BASE_MACRO_F1 = 0.424111097737` VALID; V3-A OOF publicado: `0.459793826869`.
- Los placeholders del encargo se resuelven con esa referencia, que coincide
  con HEAD y la baseline promovida del README. `git fetch origin` fue denegado
  por permisos de escritura de `.git`; no se pudo actualizar la referencia.
- No se modifica V2/V3, no se carga TEST y no se genera submission.

## Hipótesis e interfaz

El historial genera siempre ocho filas `(client_id, candidate)`, en este orden:
`cloud, gym, insurance, mobile, music, software, streaming, none`.
El target binario es exclusivamente `candidate == official TRAIN target`.
No se reconstruyen targets en pseudo-cutoffs ni se consulta una elección oracle.
Una familia sin mapping sigue siendo una candidata válida.

La implementación reutilizable vive en
`src/transaction_forecasting/ubs/soft_candidate_ranker.py`:

- `EvidenceProvider.fit(history, labels).transform(unseen_history)` devuelve
  `CandidateBatch`, incluyendo features y procedencia de cada partición.
- `cross_fitted_candidates` produce features OOF dentro de una partición TRAIN.
- `SoftCandidateRanker(kind).fit(batch, target)` aprende la relevancia oficial.
- `predict_scores` devuelve ocho logits; `predict_proba` normaliza por cliente.
- `CandidateBatch.validate` rechaza self-label provenance, clientes duplicados,
  clases ausentes/desordenadas, features no finitas e ID/target como features.

## Evidencia y normalización

72 features por candidata: probabilidades V3-A y V2, log-probabilidad e
interacciones por clase, counts/shares/aliases mapeados, máxima evidencia textual,
masa soft, recurrencia máxima/acumulada, streams supporting/due, regularidad,
estabilidad del importe, recencia, próximo evento plausible y ausencia de mapping.
El contexto incluye entropía, gap entre familias, número de familias con evidencia,
recencia/span/tamaño del historial, probabilidad baseline de `none` e interacciones
del contexto con la candidata `none`.

El mapping supervisado aprende presencia por cliente y descripción exacta:
`log(((n_family + 1)/(N_family + 2)) /
((n_other + 1)/(N_other + 2)))`.
La masa soft es `softmax(seven_log_lifts) * support/(support+5)`.
Son asociaciones débiles con el target por cliente, no probabilidades calibradas
de etiquetas de eventos. El mapping hard, usado únicamente como feature, requiere
soporte >=5 y lift >=1.5. Descripciones desconocidas tienen masa cero y un indicador.

Los streams usan pagos salientes de tarjeta y clave
`(client_id, description, currency)`; no se mezclan importes de monedas diferentes.
La utilidad heredada colapsa timestamps repetidos dentro de cada stream; sus
counts representan eventos con timestamp distinto.
Las features baseline y el contexto histórico conservan el resto del historial.
`exp(-sum_recurrence)` es un proxy explícitamente no calibrado de ausencia de
recurrencia, no una estimación validada de supervivencia.

Comparador raw family: logits positivos = `log1p(soft_recurrence_sum) + max_log_lift`,
logit de `none` = proxy anterior; softmax sobre ocho logits.
Ambos scorers aprendidos usan **softmax de logits binarios por cliente, T=1**,
sin ajustar prevalencias, sesgos por clase ni temperatura. No se promete calibración.
La baseline conserva íntegra su receta histórica, incluida la heurística fija.

## Protocolo de selección congelado antes de VALID

Cinco folds externos estratificados por cliente, seed 42. Dentro de cada TRAIN
externo se generan tres folds de probabilidades baseline y mappings soft. Cada
baseline V3-A utiliza además sus cinco folds internos de family features.
El fold externo completo queda excluido de todos los generadores internos.
No se corta una matriz OOF global para entrenar/evaluar el meta-modelo.

Comparación limitada a dos modelos:

1. `StandardScaler + LogisticRegression(C=0.1, max_iter=1500)`.
2. CatBoost binario, 300 iteraciones, depth 4, learning rate 0.05, L2=5.

Cada cliente tiene peso inverso a la frecuencia de su target TRAIN, calculado
solo dentro del fit del scorer; las ocho filas reciben el mismo peso.
CatBoost se selecciona únicamente si cumple simultáneamente:

- Macro-F1 OOF agregado >= logística + 0.005;
- gana a logística en al menos 4/5 folds;
- Macro-F1 bajo corrupción >= logística - 0.005;
- F1 `none` OOF >= logística - 0.01.

En caso contrario se conserva logística como resultado experimental. Esta regla
no implica promover el ranker por encima de la baseline.

Corrupción: eliminación aleatoria del 25% de eventos, seed 2026, conservando el
primer evento de cada cliente. Se vuelve a calcular toda la evidencia y baseline
del holdout degradado con el mismo provider; no se reentrena ni cambia el target.
Mide sensibilidad a historial incompleto, no rendimiento sobre otro target.

Segmentos definidos antes de evaluar: 0/1/>1 familias con count mapeado;
especificidad textual alta si max log-lift >= log(3); confianza recurrente alta
si max recurrence score >=1. Se reportan ambas mitades sin optimizar umbrales.
Rank accuracy equivale a accuracy del argmax; top-2 usa orden estable de clases.

VALID se carga para inferencia después de guardar `frozen_selection.json`.
Sus labels solo se abren después de ajustar todos los modelos y guardar las
predicciones. La selección queda enlazada a hashes de fuente/TRAIN y del summary
OOF. La validation oficial ya se había reutilizado en trabajos anteriores: este
resultado no es un holdout históricamente independiente.

## Reproducción y artefactos

```powershell
.venv/Scripts/python.exe scripts/experiments/v4_soft_candidate_santiago.py --phase oof
.venv/Scripts/python.exe scripts/experiments/v4_soft_candidate_santiago.py --phase valid
```

`--phase all` realiza ambas fases en orden. Si cambia fuente o TRAIN, usar un
directorio nuevo con `--output-dir`; las particiones terminadas pueden reanudarse
sin cambios. El runner rechaza volver a evaluar VALID en el mismo directorio.

Directorio local ignorado: `outputs/metrics/v4_soft_candidate_santiago/`.
Conserva matrices de features y procedencia, probabilidades/scores/predicciones
por fold, OOF alineado por `client_id`, corrupción, freeze y métricas segmentadas.
Archivos de integración: `train_oof_probabilities.csv`, `valid_probabilities.csv`,
`train_oof_selected_predictions.csv`, `valid_selected_predictions.csv`,
`summary.json` (OOF inmutable), `valid_results.json` y `final_summary.json`.

## Limitaciones predeclaradas

El mapping sigue dependiendo de descripciones exactas. El protocolo anidado
entrena generadores internos con menos clientes que el provider final; puede
haber cambio de distribución de features al reajustar con todo TRAIN.
El scorer final usa los cinco bloques OOF externos como features de entrenamiento.
La comparación entre dos recetas y sus segmentos reutiliza TRAIN OOF para
selección; no constituye una evaluación independiente de la elección de receta.
No se ejecutan ranking loss, búsqueda masiva ni ensemble posterior.

## Verificación y estado Git

- 98 tests del repositorio pasan con Python 3.12; incluye 10 casos específicos
  del ranker, ambos scorers y equivalencia con la baseline A de referencia.
- `ruff check .` pasa, con avisos por temporales inaccesibles ajenos al cambio.
- Lint y formato sobre la lista explícita de 67 archivos Python pasan.
  `ruff format --check .` falla al recorrer esos temporales; no falla sobre las fuentes.
- `pre-commit run --all-files` se intentó y quedó bloqueado al descargar hooks
  desde GitHub por la red restringida. No se declara superado.
- Imports, hashes de fuente/TRAIN y freeze inmutable verificados al terminar.
- CSV seleccionado OOF: 2.000 clientes, ocho probabilidades; 16.000 candidatos.
  CSV VALID: 1.000 clientes, ocho probabilidades; 8.000 candidatos. IDs exactos,
  orden de clases, scores finitos, normalización y argmax/predicción comprobados.
- Summary agregado, sin IDs de clientes ni datos: `v4_soft_candidate_santiago_summary.json`.
  Scores, probabilidades, matrices y procedencia por cliente permanecen ignorados.
- Rama comprobada al final: `exp/v4-soft-candidate-santiago`. No se hizo commit
  ni push: `git add` fue denegado al crear `.git/index.lock`. `.git` es de solo
  lectura para este entorno; los cinco archivos del cambio quedan sin staging.
