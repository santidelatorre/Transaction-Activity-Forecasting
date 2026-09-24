# Team handoff — Text / Descriptions / Merchants

## 1. Scope

Javier, persona 2 de 7. Se investigó normalización de `description`, TF-IDF de
palabras y caracteres, términos discriminativos, diversidad/concentración de
descripciones, frecuencia y asociaciones por clase, y dos blends de peso fijo
con la heurística V1. El contrato UBS no incluye `merchant_id` ni columna
`merchant`: una descripción normalizada es sólo una aproximación de comercio.
Quedaron fuera modelos de producción, refit sobre train+valid, nuevas
submissions y etiquetas de cada transacción individual.

## 2. Baseline

V1 oficial: Macro-F1 **0.2710243**, accuracy **0.2660**, 1.000 clientes
de validación y submission válida. Se entrenó con los 2.000 clientes oficiales
de train, se evaluó con el split oficial de validación, target
`target_next_recurring_merchant`, corte `2026-01-01` y las ocho clases fijas.
La mejor V1 fue `recurrence_heuristic`, calibrada en esa validación con
`none_bias=-1.0` y `temperature=1.0`. El runner experimental lee las
predicciones V1 por `client_id`, comprueba las etiquetas y reproduce su F1.
No se usaron etiquetas de test.

## 3. Hypotheses investigated

1. **Normalización:** quitar acentos/ruido y sustituir fechas, referencias y
   números debería unir variantes del mismo comercio. Se comparó word TF-IDF
   crudo y normalizado; el normalizador es determinista y sólo ve historias.
2. **TF-IDF word:** unigramas/bigramas de las descripciones históricas podrían
   distinguir familias. Vocabulario ajustado exclusivamente con train;
   `min_df=2`, `max_df=0.98`, máximo 2.500 términos.
3. **TF-IDF char:** n-grams `char_wb` 3–5 podrían resistir faltas y variantes.
   Vocabulario ajustado exclusivamente con train, máximo 3.500 términos.
4. **Merchant proxy:** frecuencia, número de descripciones únicas, entropía,
   concentración, recurrencia, longitud, tokens y dígitos podrían capturar
   señales que el documento TF-IDF pierde. Tablas de frecuencia y asociación
   por clase ajustadas sólo en train; al transformar train se resta el propio
   cliente de las estadísticas supervisadas.
5. **Términos discriminativos:** presencia de términos dentro de cada
   descripción podría explicar el modelo. Lift por cliente calculado sólo en
   train, con soporte mínimo de 20 clientes de la clase; se evitan bigramas
   entre transacciones distintas.
6. **Complementariedad con V1:** un blend fijo 75% heurística / 25% modelo
   textual o merchant podría recuperar `none` sin perder familias. El peso
   se fijó antes de medir esos dos blends; no se ajustó por clase.

## 4. Experiments performed

Todos los modelos ML usan los 218 agregados numéricos V1 como base, regresión
logística (`C=0.3`, `class_weight=balanced`, `tol=1e-5`, semilla 42) y
transformaciones entrenadas sólo con train. Los tiempos de entrenamiento del
clasificador están en `docs/UBS_TEXT_V2.md`; las métricas completas y matrices
de confusión se regeneran en `outputs/metrics/ubs_text_v2/`.

| experiment_id | change | macro_f1 | delta_vs_v1 | accuracy | affected_classes | result |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `v1_selected` | Heurística V1 elegida | 0.271024 | 0 | 0.266 | Referencia | Baseline |
| `numeric_control` | Agregados V1 sin texto | 0.199439 | -0.071585 | 0.273 | `none` ↑; 7 familias ↓ | Inferior |
| `v1_word_component` | TF-IDF crudo word 1–2 | 0.200045 | -0.070979 | 0.280 | `none` ↑; 7 familias ↓ | Inferior |
| `normalized_unigram` | Normalización + unigramas | 0.197970 | -0.073054 | 0.281 | `none` ↑; 7 familias ↓ | Inferior |
| `normalized_word12` | Normalización + word 1–2 | 0.200045 | -0.070979 | 0.280 | `none` ↑; 7 familias ↓ | Igual al componente word crudo |
| `normalized_char` | Normalización + char 3–5 | 0.196875 | -0.074149 | 0.281 | `none` ↑; 7 familias ↓ | Inferior |
| `merchant_proxy` | 27 estadísticas de descripción | 0.169151 | -0.101873 | 0.285 | `none` ↑; 7 familias ↓ | Inferior |
| `text_plus_merchant` | Word 1–2 + merchant proxy | 0.169339 | -0.101685 | 0.287 | `none` ↑; 7 familias ↓ | Inferior |
| `v1_plus_word_blend` | 75% V1 + 25% word | 0.232839 | -0.038186 | 0.276 | `cloud`, `none` ↑; otras 6 ↓ | Inferior |
| `v1_plus_merchant_blend` | 75% V1 + 25% merchant | **0.243527** | **-0.027497** | **0.300** | `cloud`, `none` ↑; otras 6 ↓ | Mejor variante nueva, inferior a V1 |

El archivo `experiments.json` registra `experiment_id`, descripción, cambio,
Macro-F1, delta, accuracy, F1 por clase, clases afectadas, commit, notas,
resultado, tiempo y dimensiones. El blend y la elección de la mejor
representación comparten validación con la V1 ya calibrada; los resultados
son exploratorios, no una estimación independiente de generalización.

## 5. What worked

- El texto contiene señales legibles por familia: por ejemplo, lift en train
  de `cloud access` 4.44×, `fitness monthly` 4.28×, `safe cover` 4.43×,
  `phone contract` 4.28×, `member pass` 4.20×, `productivity suite` 4.02×,
  `video access` 4.03× y `member plan` para `none` 6.00×.
- El modelo word 1–2 eleva F1 de `none` de 0.0942 a 0.3891, pero su
  Macro-F1 global es 0.200045. Es una señal útil para analizar errores,
  no una mejora del clasificador completo.
- El blend V1 + merchant proxy sube F1 de `none` a 0.3969 y de `cloud` a
  0.3143; también sube accuracy a 0.300. Pierde en las otras seis familias,
  por lo que tampoco mejora la métrica objetivo.

## 6. What did NOT work

- Ninguna variante superó la Macro-F1 0.2710243 de V1.
- Word 1–2 normalizado dio exactamente 0.200045, igual que el componente
  word crudo; el lector V1 ya pasa texto a minúsculas y la normalización
  adicional no produjo vocabulario útil distinto en esta muestra.
- Character TF-IDF obtuvo 0.196875; no compensó la pérdida de familias.
- Merchant proxy solo dio 0.169151 y combinado con word 0.169339. La
  frecuencia y las asociaciones exactas entre descripciones no generalizaron
  suficientemente al split oficial.
- Los blends fijos mejoraron frente al ML aislado, pero quedaron 0.0275–0.0382
  puntos por debajo de V1. No deben convertirse en predictor V2 por su mayor
  accuracy.

## 7. Main findings

1. La V1 heurística sigue siendo el mejor resultado en Macro-F1.
2. `none` es el principal punto débil de V1 (F1 0.0942); el texto lo detecta
   mejor, con un coste alto para las familias.
3. Un documento único por cliente mezcla muchos comercios; puede ser demasiado
   grueso para identificar el *próximo* recurrente.
4. Train tiene mediana de 24–26 descripciones distintas por cliente, validación
   30–36; la entropía mediana aumenta de ≈3.0 a 3.2–3.4. Hay cambio de
   distribución relevante para features de frecuencia.
5. El dataset sintético contiene palabras cercanas a las etiquetas. Su lift
   en train es explicativo, pero podría no trasladarse a otro generador.
6. En análisis de términos, unir transacciones en un documento crea bigramas
   espurios en los límites; contar presencia dentro de cada descripción los evita.

## 8. Recommendations for V2

- **MUST:** conservar V1 como baseline y seleccionar con Macro-F1 de ocho
  clases, F1 por clase y el split oficial.
- **MUST:** ajustar vocabularios, escalados y asociaciones sólo con train;
  mantener exclusión del propio cliente en estadísticas supervisadas.
- **SHOULD:** investigar texto a nivel de secuencia recurrente o candidato,
  vinculado con recencia y regularidad, antes de integrarlo en el clasificador.
- **COULD:** usar términos discriminativos y estadísticas de diversidad para
  explicar errores o como señales candidatas en un experimento nuevo.
- **AVOID:** sustituir V1 por cualquiera de los clasificadores o blends aquí
  probados; todos pierden Macro-F1.
- **AVOID:** llamar `merchant` a la descripción como si fuera un identificador
  limpio o usar la etiqueta del cliente como etiqueta de cada transacción.

## 9. Code worth integrating

| Ruta | Función | Dependencia | Estado |
| --- | --- | --- | --- |
| `src/transaction_forecasting/ubs/text_v2.py` | Normalización, documentos y features de merchant proxy con leave-one-client-out | pandas, numpy y contrato UBS | Reutilizable para investigación; no seleccionar como predictor actual |
| `scripts/run_ubs_text_v2.py` | Ablaciones, métricas, términos, matrices y tracking | V1, sklearn, scipy y artefacto de validación V1 | Reproducible como runner experimental |
| `configs/ubs_text_v2.toml` | Parámetros, seed y blend fijo | Runner V2 | Listo para repetir experimentos |
| `tests/test_ubs_text_v2.py` | Normalización y ausencia de autocontribución supervisada | pytest | Listo para integrar con utilidades |
| `docs/UBS_TEXT_V2.md` | Métricas detalladas y reproducción | Ninguna | Referencia de resultados, no código de producción |

## 10. Commits worth integrating

- `f598fc3` — Investigación inicial, módulo de texto, runner, configuración,
  tests y resultados. **CHERRY-PICK RECOMMENDED** para reutilizar la
  infraestructura experimental; no activar sus modelos como reemplazo de V1.
- `7ef31a6` — Blends fijos con V1 y tracking estructurado. **CHERRY-PICK
  RECOMMENDED** después de `f598fc3` si se desea reproducir la comparación.
  Ambos commits dejan V1 intacta.

## 11. Dependencies / conflicts

No se añadió librería: pandas, numpy, scipy y scikit-learn ya están en el
proyecto. El runner importa el `ClientFeatureBuilder`, `RecurrenceHeuristic`,
contrato de datos y evaluador UBS existentes. Requiere ejecutar V1 primero
para disponer de `outputs/metrics/ubs_v1/validation_error_analysis.csv`; el
artefacto está ignorado por Git. Puede solaparse conceptualmente con el
workstream de feature engineering o modelos, pero no edita sus módulos. La
integración debe contrastar cualquier nueva señal con el equipo temporal y
el responsable de validación.

## 12. Risks

- **Leakage:** las asociaciones por clase del target cliente no son etiquetas
  por transacción. La implementación resta el cliente propio en train y no
  toca etiquetas de validación; cualquier refactor debe preservar esto.
- **Sobreajuste:** vocabularios y descripciones de alta cardinalidad producen
  asociaciones frágiles. Seleccionar la mejor variante o el blend en la
  misma validación reutilizada por V1 sesga la estimación.
- **Cambio de distribución:** validación tiene más descripciones distintas y
  mayor entropía por cliente que train.
- **Generador sintético:** términos como `cloud access` pueden revelar pistas
  literales no disponibles en producción real.
- **Coste:** el clasificador suele tardar menos de 3 s, pero el coste completo
  incluye carga y construcción de features; la V1 completa tardó ≈171 s en
  la ejecución observada.
- **Dependencia de artefactos:** el runner V2 exige el archivo de predicciones
  de validación de V1 y comprueba IDs, etiquetas y Macro-F1.

## 13. Recommended next experiment

**Una sola prueba:** construir documentos únicamente para las descripciones
de cada secuencia recurrente detectada antes del corte, añadir TF-IDF word 1–2
al score de recencia/regularidad de V1 y comparar con la heurística V1.
Ajustar vocabulario y cualquier peso con train mediante folds disjuntos por
cliente; evaluar una vez en validación oficial. Criterio de éxito: Macro-F1
superior a 0.2710243 sin caída mayor de 0.03 en ninguna de las siete familias
frente a V1. No usar etiqueta de validación en features ni en selección de
pesos; tratar el resultado como exploratorio por la calibración previa de V1.

## 14. Executive summary for integration AI

- La mejor V1 sigue en Macro-F1 0.2710243; ninguna variante de Javier la supera.
- Mejor variante nueva: blend fijo 75% V1 + 25% merchant proxy, 0.243527
  (delta -0.027497), accuracy 0.300.
- Texto y merchant proxy elevan F1 de `none`, pero perjudican la mayoría de familias.
- Word normalizado iguala al componente word crudo; char TF-IDF y diversidad
  tampoco justifican integración predictiva.
- No existe merchant ID: `description` es sólo aproximación.
- Conservar V1. Investigar texto por secuencia recurrente como siguiente prueba.
- Revisar `f598fc3` y después `7ef31a6` sólo por utilidades experimentales.
- Leer `docs/UBS_TEXT_V2.md` y regenerar `experiments.json` para detalle.
