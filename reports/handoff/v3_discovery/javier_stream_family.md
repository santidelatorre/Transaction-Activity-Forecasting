# Javier — identificación de familia por description/stream

## Resumen

El experimento separa la clasificación de las siete familias positivas de la decisión
`none`. El mejor modelo seleccionado exclusivamente por Macro-F1 OOF en TRAIN es TF-IDF
de caracteres con regresión logística regularizada. Obtiene **0.448772 Macro-F1 OOF** y
**0.359701 Macro-F1 en VALID** sobre clientes cuya clase real es positiva. En VALID,
`music` llega a **0.368159 F1** y `streaming` a **0.272727 F1**.

La señal textual existe, pero el label a nivel cliente limita la calidad de los candidatos:
un 14.4% de los clientes positivos de TRAIN no tiene ningún stream que supere el filtro
leave-one-client-out de soporte y lift. La regla exacta de aliases supera en VALID al modelo
seleccionado (0.378677), pero no fue la ganadora OOF y se conserva solo como diagnóstico.

La decisión `none` presenta una deriva grave. Su gate binario pasa de F1 0.686688 OOF a
0.454976 en VALID y predice `none` para 973 de 1.000 clientes. Por ello no debe sustituir
la decisión `none` de V2.

## Protocolo y prevención de leakage

- Baseline congelada: `5884ddd07a19b2e19b44b4fd55afdf4eb6dcf749`.
- Cinco folds estratificados por `client_id`, semilla 42. Como existe una fila de label por
  cliente, los folds son simultáneamente folds de grupo; ningún cliente aparece en fit y
  holdout.
- Los modelos y mappings de cada fold se ajustan solo con los clientes fit de TRAIN.
- VALID se usa solo después de seleccionar la configuración por OOF: fit final en TRAIN e
  inferencia/scoring en VALID.
- No se propaga el target del cliente a todos sus streams. Se elige como máximo un stream
  candidato por cliente positivo.
- Para seleccionar el candidato de un cliente se resta primero su propia contribución de
  los conteos description-clase. El score combina lift de alias, soporte, recurrencia,
  periodicidad, estabilidad de amount y recencia. El lookup que decide el candidato no
  puede beneficiarse del propio label.
- `none` no se usa como weak label de ningún stream. Se estudia con un clasificador binario
  separado a nivel cliente.

En el fit final hay 1.403 candidatos para 1.403 clientes positivos. El 85.60% supera el
filtro de soporte/lift y el 14.40% usa el mejor fallback disponible. La mediana del candidato
es cuatro apariciones, soporte leave-one-out 455 y log-lift 1.491. Esto sigue siendo weak
supervision: no convierte el candidato en ground truth de stream.

## Comparación de modelos

Las métricas siguientes se calculan solo sobre las siete familias positivas. La selección
se hace por la columna OOF, no por VALID.

| Configuración | Macro-F1 OOF | Macro-F1 VALID | Accuracy VALID |
| --- | ---: | ---: | ---: |
| TF-IDF char, seleccionado | **0.448772** | 0.359701 | 0.371994 |
| TF-IDF word | 0.447981 | 0.358318 | 0.369165 |
| texto + periodicidad | 0.447794 | 0.356552 | 0.363508 |
| texto + otras señales | 0.446404 | 0.336482 | 0.347949 |
| reglas exactas de alias | 0.443820 | **0.378677** | **0.386139** |
| texto + amount stability | 0.437851 | 0.333920 | 0.340877 |
| texto + MCC/type/direction + señales | 0.434763 | 0.323931 | 0.332390 |
| TF-IDF word + char | 0.431044 | 0.343782 | 0.356436 |
| texto + MCC/type/direction | 0.429025 | 0.334521 | 0.350778 |

El ranking es muy plano en OOF entre char, word y texto+periodicidad. Añadir MCC, type,
direction, periodicidad, estabilidad de amount u otras señales no mejora al texto solo. La
caída de 0.448772 OOF a 0.359701 VALID indica drift de asociaciones, ruido del candidato o
ambos; no es razonable hacer tuning más fino con este split.

## Resultado por clase

Modelo seleccionado, familias positivas:

| Clase | F1 OOF | Precision VALID | Recall VALID | F1 VALID |
| --- | ---: | ---: | ---: | ---: |
| cloud | 0.478351 | 0.335025 | 0.741573 | 0.461538 |
| gym | 0.488038 | 0.466667 | 0.404959 | 0.433628 |
| insurance | 0.459821 | 0.318681 | 0.292929 | 0.305263 |
| mobile | 0.410557 | 0.425287 | 0.355769 | 0.387435 |
| music | 0.430000 | 0.342593 | 0.397849 | **0.368159** |
| software | 0.443787 | 0.387097 | 0.230769 | 0.289157 |
| streaming | 0.430851 | 0.368421 | 0.216495 | **0.272727** |

### Confusion matrix VALID, familias positivas

| real / pred | cloud | gym | insurance | mobile | music | software | streaming |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 66 | 4 | 4 | 4 | 6 | 2 | 3 |
| gym | 27 | 49 | 10 | 12 | 14 | 6 | 3 |
| insurance | 22 | 10 | 29 | 6 | 13 | 11 | 8 |
| mobile | 21 | 4 | 16 | 37 | 14 | 4 | 8 |
| music | 14 | 11 | 13 | 5 | 37 | 7 | 6 |
| software | 23 | 14 | 12 | 10 | 13 | 24 | 8 |
| streaming | 24 | 13 | 7 | 13 | 11 | 8 | 21 |

El error dominante global es la sobrepredicción de `cloud`. En music, los errores más
frecuentes son `cloud` (14), `insurance` (13) y `gym` (11). En streaming son `cloud` (24),
`gym` (13) y `mobile` (13).

## Music frente a streaming

Hay **6 errores music → streaming** y **11 streaming → music**. La separación directa no
es el mayor par de confusión: ambas clases se pierden más a menudo hacia familias ajenas,
especialmente cloud. El vocabulario central sí sigue una semántica clara:
`audio streaming` y `member pass` se asocian a music; `media streaming` y `video access`, a
streaming. Sin embargo, sus confianzas de alias están entre 0.32 y 0.36 porque los mismos
aliases aparecen también en clientes con otros targets.

Frente a V2 en el split oficial, el clasificador aislado eleva music de aproximadamente
0.17 a 0.368, mientras streaming solo sube de 0.25 a 0.273. Esto demuestra valor para
music, pero no una solución completa para la familia.

## Aliases aprendidos

La confianza es `(soporte_familia + 1) / (soporte_total + 8)` sobre presencia por cliente.
No es una probabilidad calibrada de que el stream sea el evento target.

| Descripción/alias | Familia aprendida | Soporte | Confianza |
| --- | --- | ---: | ---: |
| cloud access | cloud | 441 | 0.342984 |
| storage plan | cloud | 445 | 0.315673 |
| gym membership | gym | 483 | 0.297352 |
| urban gym | gym | 433 | 0.328798 |
| insurance monthly | insurance | 455 | 0.362851 |
| safe cover | insurance | 449 | 0.369803 |
| phone contract | mobile | 474 | 0.329876 |
| service bill | mobile | 464 | 0.317797 |
| member pass | music | 449 | 0.332604 |
| audio streaming | music | 456 | 0.321121 |
| saas billing | software | 452 | 0.321739 |
| productivity suite | software | 464 | 0.313559 |
| media streaming | streaming | 494 | 0.344622 |
| video access | streaming | 482 | 0.355102 |
| digital plus | mobile | 1.063 | 0.154062 |
| monthly plan | mobile | 762 | 0.203896 |
| premium plan | streaming | 1.083 | 0.172319 |

Los aliases genéricos `digital plus`, `monthly plan` y `premium plan` tienen mucho soporte
y poca pureza. No deben convertirse en reglas duras. El vocabulario parece sintético o
plantillado: hay pocas raíces semánticas, sufijos repetidos (`online`, `digital`, `plus`,
`service`, `core`) y variantes/abreviaturas sistemáticas. Las descripciones semánticas son
útiles, pero su asociación con el target de cliente no equivale a un label de stream.

## Coverage y evidencia

- TRAIN contiene 1.475 descripciones normalizadas; los streams recurrentes/fallback de
  VALID contienen 240.
- 228/240 descripciones únicas de VALID se ven en TRAIN. Las 12 no vistas representan
  solo 0.0561% de las apariciones ponderadas.
- El 99.9272% de las filas de stream y el 99.9439% de las apariciones de VALID usan una
  descripción conocida.
- Todos los streams ganadores de los 707 clientes positivos usan una descripción conocida.
  La generalización a descripciones realmente nuevas no se puede estimar con este split.

| Apariciones del stream ganador | Clientes | Accuracy | Macro-F1 | Score medio |
| --- | ---: | ---: | ---: | ---: |
| 2 | 449 | 0.363029 | 0.354433 | 0.904995 |
| 3–4 | 211 | 0.374408 | 0.357303 | 0.846923 |
| 5+ | 47 | **0.446809** | **0.389002** | 0.781739 |

Más evidencia mejora el resultado observado, aunque el grupo 5+ es pequeño. El score del
modelo es más alto con dos apariciones y por tanto está mal calibrado como confianza. Solo
debe usarse para ranking y filtros conservadores.

## Decisión `none`

El gate binario obtiene en TRAIN OOF precision 0.666142, recall 0.708543 y F1 0.686688.
En VALID cae a precision 0.295992, recall 0.982935 y F1 0.454976, con 973 predicciones
`none`. El pipeline combinado baja a Macro-F1 0.074278 en VALID. Esta capa se rechaza.

El problema no es falta de coverage de aliases. Hay un cambio fuerte en la relación entre
historial textual y `none`, y la clasificación de familias positivas no debe asumir que su
ausencia de confianza resuelve `none`.

## Integración mínima exploratoria con V2

Se preserva siempre la decisión `none` de V2. Con umbral fijo 0.85 sobre el score del modelo
seleccionado:

| Corrección | Elegibles | Cambios | Macro-F1 ocho clases |
| --- | ---: | ---: | ---: |
| V2 sin corrección | — | — | 0.391549 |
| corregir cualquier familia positiva | 603 | 269 | 0.394246 |
| corregir solo hacia music/streaming | 131 | 94 | **0.412915** |

La corrección focal eleva music de 0.169935 a 0.333333 y streaming de 0.250000 a 0.319588,
manteniendo none en 0.519380. También reduce insurance, mobile y software. Esta comparación
es exploratoria y se ha medido sobre el VALID ya reutilizado; no se recomienda congelarla
sin un holdout nuevo o evidencia OOF comparable de V2.

## Reproducción y artefactos

```powershell
.\.venv\runtime_v2\python.exe scripts/experiments/javier_stream_family.py
```

El script genera en `outputs/metrics/v3_discovery/javier_stream_family/`:

- `results.json`, protocolo y métricas completas;
- `model_comparison.csv` y matrices de confusión;
- `learned_aliases.csv`, tabla completa de aliases;
- `coverage.json` y `evidence_bands.csv`;
- `valid_predictions.csv`, diagnóstico local ignorado por Git.

## Conclusión

La clasificación aislada aporta una señal clara para music y una mejora exploratoria al
corregir V2 solo hacia music/streaming. El gap OOF→VALID, la baja pureza de los aliases, el
fallo de `none` y la ausencia de mejora consistente al añadir MCC/periodicidad/amount
impiden considerarla el cuello de botella principal o promover la integración sin otra
validación.

**STREAM FAMILY CLASSIFICATION HAS MODERATE VALUE**
