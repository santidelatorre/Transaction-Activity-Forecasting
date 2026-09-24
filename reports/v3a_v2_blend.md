# Experimento V2 + V3-A: selección exclusiva con TRAIN OOF

Fecha: 24 de septiembre de 2026. Rama: `experiment/v3a-v2-blend`, desde
`integration/v3-discovery` (`df9fe41`).

## Resultado y recomendación

**Ganó alpha = 1.00: V3-A puro.** TRAIN OOF Macro-F1 =
**0.459793826869**; VALID Macro-F1 = **0.424111097737**.
Ninguna mezcla interior superó a V3-A en TRAIN OOF. El experimento no aporta
un blend que mejore sobre A: la selección reproduce exactamente sus probabilidades
y predicciones. Queda **0.005790062003 por debajo** del
ensemble anterior en VALID. No se evaluaron otros pesos de V2+A en VALID.

**Recomendación: promover V3-A como candidato de esta rama, elegido por OOF;
no promover un nuevo V2+A.** Hay evidencia favorable frente a V2 en estos datos,
pero no se ha demostrado superioridad sobre V2+full ni generalización independiente.
No se cambia el alpha para perseguir el resultado de VALID. Esta recomendación
no ejecuta ningún cambio de producción, integración ni envío al challenge.

## Protocolo y secuencia verificable

- Fórmula única: `P = (1 - alpha) * P(V2) + alpha * P(V3-A)`; argmax en el
  orden oficial de ocho clases. Sin calibración, pesos por clase ni nuevas features.
- Corte `2026-01-01`, seed 42, scorer compartido
  `ubs.evaluation.evaluate_predictions` → `evaluation.official.classification_metrics`.
- Se reutilizaron los cinco folds estratificados por cliente de TRAIN (2.000
  clientes), con los mismos cinco folds internos de mappings supervisados de V3.
  Se reconstruyó la partición con `StratifiedKFold(5, shuffle=True, random_state=42)`
  y se comprobó el conjunto exacto de IDs de cada fold, sin solapamiento.
- Cada tabla se alinea por `client_id` y columnas oficiales. Se rechazan IDs
  repetidos/ausentes/extra, NaN, clases incorrectas y probabilidades inválidas.
  Se leen los floats con `float_precision="round_trip"`, sin renormalizarlos.
- Antes de buscar pesos se verificaron los hashes originales de código y TRAIN,
  las versiones, los argmax guardados y los scores de ambos extremos. La fase
  OOF no lee las etiquetas ni transacciones VALID. No reentrena V2 ni V3-A.
- Grid y desempate fijados en código antes de la búsqueda: tolerancia absoluta
  `0.0001` Macro-F1; dentro de ella, preferir modelo puro, después cercanía a
  50/50, después alpha menor. El ganador es también el máximo estricto; **no se
  activó el desempate**. La diferencia frente al mejor peso interior (0.50) es
  **0.003763818432**, superior a la tolerancia.
- Freeze creado a `2026-09-24T20:18:32.834772+00:00`; inicio de VALID a
  `2026-09-24T20:18:43.587820+00:00`; fin a
  `2026-09-24T20:18:45.163724+00:00`. El checksum es
  `8b60651a8e78d8c3fcb810f1c414651c633faa6a707a639e83ea6043ca9bb468`.
- `frozen_selection.json` registra alpha, regla, criterio, métricas OOF, inputs,
  fuentes y fingerprints. Su checksum, los archivos OOF y las fuentes se verifican
  antes de VALID y TEST. Escritura exclusiva (`x`) y marcador de evaluación
  impiden sobrescribir el freeze o repetir VALID en el mismo directorio.
- Solo se evaluó el alpha congelado sobre los 1.000 clientes VALID. Los cuatro
  controles solicitados se puntuaron en esa misma fase. Todas las predicciones
  estaban guardadas antes de decodificar las etiquetas VALID para scoring.

La hipótesis de usar A se formuló con resultados históricos de este mismo VALID.
Por tanto, **la selección del peso está aislada de VALID, pero VALID no es un
holdout independiente de toda la investigación previa**. El Macro-F1 OOF del
ganador también tiene optimismo por selección entre once pesos.

## Búsqueda completa en TRAIN OOF

| alpha (peso A) | TRAIN OOF Macro-F1 | Accuracy |
| --- | --- | --- |
| 0.00 | 0.408883545420 | 0.4230 |
| 0.10 | 0.417628165619 | 0.4315 |
| 0.20 | 0.435739710883 | 0.4485 |
| 0.30 | 0.445767764014 | 0.4575 |
| 0.40 | 0.452915330200 | 0.4645 |
| 0.50 | 0.456030008437 | 0.4690 |
| 0.60 | 0.455361430302 | 0.4680 |
| 0.70 | 0.452919955677 | 0.4650 |
| 0.80 | 0.453251995584 | 0.4655 |
| 0.90 | 0.455266480407 | 0.4665 |
| 1.00 | 0.459793826869 | 0.4710 |

El CSV de búsqueda incluye también accuracy, los ocho F1 y los ocho prediction
counts. `oof_results.json` conserva además la matriz de confusión de cada alpha.
Los resultados por fold del candidato congelado son:

| Fold TRAIN | V2 | V3-A / seleccionado | Delta vs V2 |
| --- | --- | --- | --- |
| 1 | 0.386089231 | 0.422836749 | +0.036747519 |
| 2 | 0.414130599 | 0.449722327 | +0.035591728 |
| 3 | 0.423242384 | 0.476689542 | +0.053447158 |
| 4 | 0.426194003 | 0.474124751 | +0.047930748 |
| 5 | 0.387149451 | 0.465166029 | +0.078016578 |

La mejora sobre V2 aparece en los cinco folds. Son diagnósticos del mismo OOF
usado para seleccionar, no cinco réplicas independientes ni una validación
anidada del procedimiento de selección de alpha.

## Comparación exacta

| Modelo | TRAIN OOF Macro-F1 | VALID Macro-F1 | VALID accuracy |
| --- | --- | --- | --- |
| V2 | 0.408883545420 | 0.391549455911 | 0.4240 |
| V3-A identity | 0.459793826869 | 0.424111097737 | 0.4610 |
| V3-full | 0.463395166332 | 0.398494213467 | 0.4480 |
| 50/50 V2 + V3-full | 0.451231679418 | 0.429901159739 | 0.4700 |
| V2 + V3-A, alpha=1 | 0.459793826869 | 0.424111097737 | 0.4610 |

Frente a V2, el delta absoluto en VALID es **+0.032561641826**
Macro-F1, equivalente a **3.256164 puntos porcentuales**;
el delta relativo es **8.316099%**. Accuracy pasa de
0.424 a 0.461 (+3.7 puntos porcentuales).
Frente a V3-A el delta es exactamente cero. Frente a V2+full el delta es
**-0.005790062003**. Los OOF de V3-full y del ensemble
anterior proceden del informe original, cuya fuente OOF se incluyó en el freeze.

## F1 por clase y comprobación especial de VALID

| Clase | V2 | V3-A | V3-full | V2+full | Seleccionado (A) |
| --- | --- | --- | --- | --- | --- |
| cloud | 0.450000000 | 0.481927711 | 0.440000000 | 0.496815287 | 0.481927711 |
| gym | 0.487272727 | 0.442477876 | 0.351219512 | 0.435146444 | 0.442477876 |
| insurance | 0.429149798 | 0.427272727 | 0.411764706 | 0.444444444 | 0.427272727 |
| mobile | 0.452674897 | 0.468085106 | 0.456621005 | 0.470085470 | 0.468085106 |
| music | 0.169934641 | 0.323809524 | 0.335078534 | 0.298850575 | 0.323809524 |
| software | 0.373983740 | 0.334801762 | 0.361904762 | 0.403587444 | 0.334801762 |
| streaming | 0.250000000 | 0.306569343 | 0.259541985 | 0.298507463 | 0.306569343 |
| none | 0.519379845 | 0.607944732 | 0.571823204 | 0.591772152 | 0.607944732 |

| Clase | F1 seleccionado | Delta vs V2 | Delta vs V3-A | Lectura |
| --- | --- | --- | --- | --- |
| gym | 0.442477876 | -0.044794851 | 0.000000000 | No recupera la pérdida de A |
| music | 0.323809524 | +0.153874883 | 0.000000000 | Conserva toda la mejora de A |
| streaming | 0.306569343 | +0.056569343 | 0.000000000 | Mantiene A; mejora sobre V2 |
| none | 0.607944732 | +0.088564887 | 0.000000000 | Conserva toda la mejora de A |

No se recupera gym porque el peso seleccionado de V2 es cero. Music, streaming
y none conservan exactamente el comportamiento de A, sin mejora adicional.
También persisten sus pérdidas frente a V2 en software y, ligeramente, insurance.
No se corrigió ninguna clase con reglas o pesos específicos.

### Prediction counts

| Clase | V2 | V3-A | V3-full | V2+full | Seleccionado (A) |
| --- | --- | --- | --- | --- | --- |
| cloud | 71 | 77 | 61 | 68 | 77 |
| gym | 154 | 105 | 84 | 118 | 105 |
| insurance | 148 | 121 | 71 | 108 | 121 |
| mobile | 139 | 131 | 115 | 130 | 131 |
| music | 60 | 117 | 98 | 81 | 117 |
| software | 142 | 123 | 106 | 119 | 123 |
| streaming | 63 | 40 | 34 | 37 | 40 |
| none | 223 | 286 | 431 | 339 | 286 |

### Matriz de confusión del candidato congelado

Filas reales; columnas predichas; orden oficial. Suma total: 1.000 clientes.

| Real / predicha | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 40 | 2 | 7 | 8 | 6 | 9 | 3 | 14 |
| gym | 5 | 50 | 13 | 12 | 9 | 11 | 2 | 19 |
| insurance | 5 | 5 | 47 | 5 | 8 | 13 | 1 | 15 |
| mobile | 4 | 5 | 12 | 55 | 6 | 7 | 0 | 15 |
| music | 4 | 8 | 12 | 8 | 34 | 11 | 2 | 14 |
| software | 7 | 8 | 7 | 11 | 8 | 38 | 2 | 23 |
| streaming | 5 | 10 | 7 | 8 | 24 | 12 | 21 | 10 |
| none | 7 | 17 | 16 | 24 | 22 | 22 | 9 | 176 |

Las matrices de los cuatro controles están en `valid_results.json`.

## Bootstrap emparejado por cliente

Se reutiliza directamente `bootstrap_delta` de `scripts/run_ubs_v3.py`.
Cada resample extrae 1.000 clientes con reemplazo y aplica los mismos índices a
verdad, control y candidato; calcula la diferencia de Macro-F1 sobre las ocho
clases fijas. El delta central es el observado en la muestra original. Se usan
2.000 resamples, `numpy.random.default_rng(42)` y percentiles 2.5/97.5.

| Comparación | Delta central | Percentil 2.5% | Percentil 97.5% | Resamples | Seed |
| --- | --- | --- | --- | --- | --- |
| Seleccionado − V2 | +0.032561641826 | +0.003638877178 | +0.061844592860 | 2000 | 42 |
| Seleccionado − V3-A identity | +0.000000000000 | +0.000000000000 | +0.000000000000 | 2000 | 42 |
| Seleccionado − 50/50 V2 + V3-full | -0.005790062003 | -0.027675803207 | +0.016302067200 | 2000 | 42 |

La mejora frente a V2 tiene un intervalo descriptivo completamente positivo y
consistencia de signo en los cinco folds OOF. Es una señal favorable, condicionada
a estos datos y a las predicciones fijadas. No demuestra robustez entre nuevos
datasets, seeds o cohortes ni elimina el efecto del uso histórico de VALID.

**Frente a V3-A no existe una mejora compatible o incompatible con ruido: son
el mismo predictor.** El intervalo `[0, 0]` es una consecuencia algebraica de
alpha=1, no una estimación de incertidumbre nula sobre el rendimiento futuro.
La diferencia negativa frente al ensemble anterior tiene un intervalo que cruza
cero; no establece una diferencia concluyente en ninguna dirección.

Estos intervalos **no son una corrección formal por selección múltiple**,
no repiten entrenamiento ni selección dentro de cada bootstrap y no corrigen
el optimismo de OOF ni la reutilización histórica de VALID.

## Submission TEST

Se volvió a entrenar la API sin cambios `V3Model.fit` con TRAIN+VALID (3.000
clientes), siguiendo el procedimiento de submission original. La API existente
ajusta internamente todos sus brazos; esta ejecución solo consume las
probabilidades V2 y A. El seed sigue siendo 42 y no se cambia la receta.
Las etiquetas VALID entran en entrenamiento **únicamente en este refit final**,
después de congelar y evaluar el candidato; no intervienen en la selección.

El máximo error absoluto frente a las probabilidades TEST originales es
V2 = `0.0` y
A = `0.0`; los argmax son
idénticos. Se aplica alpha=1 congelado y el CSV coincide con el candidato A.

Archivo: `outputs/metrics/v3a_v2_blend/submission_v3a_v2_blend.csv`.
SHA-256: `6b2420f879e1ae84987e7f3e0ee8c148d0d752982d7ba9a0e66f47b7cf9f011c`.

Validadores UBS, contrato oficial e `inspect_submission`: PASS. También pasan
los dos CLI oficiales existentes. Exactamente 1.000 filas, 1.000 IDs únicos,
mismas IDs y mismo orden que `sample_submission.csv`, esquema de dos columnas,
solo las ocho clases permitidas, cero IDs faltantes/extra/repetidos y cero nulls
o predicciones vacías. **No se ha enviado al challenge.**

## Reproducción, artefactos y verificación

Implementación: `scripts/run_v3a_v2_blend.py`, helper de combinación puro
`src/transaction_forecasting/evaluation/probability_blend.py` y
`tests/test_v3a_v2_blend.py`. V1, V2, V3 y el scorer original no se modificaron.

Desde la raíz, con los artefactos V3 originales verificados y su entorno:

```powershell
python scripts/run_v3a_v2_blend.py --phase oof --output-dir outputs/metrics/v3a_v2_blend_reproduction
python scripts/run_v3a_v2_blend.py --phase valid --output-dir outputs/metrics/v3a_v2_blend_reproduction
python scripts/run_v3a_v2_blend.py --phase submission --output-dir outputs/metrics/v3a_v2_blend_reproduction
```

El directorio OOF debe ser nuevo; las fases siguientes usan ese mismo freeze.
La ejecución registrada usó `.venv/runtime_v3/python.exe` (Python 3.12.10 x64),
NumPy 2.3.5, pandas 2.3.3, scikit-learn 1.9.1 y CatBoost 1.2.10.
En este runtime embebido se necesita el bootstrap:

```powershell
.venv/runtime_v3/python.exe -c "import sys,runpy; sys.path.insert(0,'scripts'); sys.argv=['scripts/run_v3a_v2_blend.py','--phase','oof','--output-dir','outputs/metrics/v3a_v2_blend_reproduction']; runpy.run_path(sys.argv[0],run_name='__main__')"
```

Sustituir únicamente la fase por `valid` y `submission` en las invocaciones
siguientes. Los hashes son de bytes y los directorios se fijan en el freeze;
un cambio de fuentes, entorno o datos se rechaza, no se acepta una caché obsoleta.
La caché V3 original registraba hashes de fuentes y datos, pero no de cada CSV
de probabilidades: se comprobó su coherencia con folds, IDs, predicciones y
métricas originales, y se fijaron ahora hashes de todos los CSV consumidos.
Esto es auditoría de reutilización, no una segunda ejecución del OOF completo.

Todos los resultados permanecen locales e ignorados por Git bajo
`outputs/metrics/v3a_v2_blend/`: `oof_weight_search.csv`, `oof_results.json`,
`frozen_selection.json` y su checksum, `valid_results.json`,
`oof_predictions.csv`, `valid_predictions.csv`, las probabilidades OOF/VALID/TEST,
`valid_official.csv`, submission, marcadores y provenance de VALID/TEST.
`verification.json` registra checks finales y hashes de artefactos.

- Nuevos tests: **19 passed**; suite completa, incluidos UBS y validadores:
  **106 passed** (`--basetemp=.pytest_review_tmp/v3a_blend_full`).
- `ruff check .`: PASS; `ruff format --check .`: PASS, 67 archivos.
- `pre-commit run --all-files`: PASS en los cinco hooks; los nuevos archivos
  también pasan `pre-commit run --files ...`.
- Revisión final de hashes: **81 artefactos anteriores
  intactos**, freeze intacto y fuentes originales sin cambios.
- No se incorporaron datasets, credenciales ni outputs a Git; se conservaron
  los cambios ajenos existentes. No hubo merge a main.

La pregunta del experimento se responde negativamente: **no se ha encontrado
una combinación V2+A que supere reproduciblemente a V2, A y V2+full**. El
procedimiento limpio conserva la mejora de A frente a V2, pero no la amplía.
