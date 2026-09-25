CURRENT_REPRODUCED_VALID_MACRO_F1: 0.6194934236214421

# Stream Identity: reconstrucción TRAIN-only

## Estado inicial preservado

- Rama verificada: `research/import-v2-stream-identity`.
- HEAD inicial: `d5dddfd8d93df61d3a6421f860428bf491c995f7`.
- Sin modificaciones versionadas. Directorios preexistentes sin seguimiento:
  `.pre-commit-cache/` y `.runtime314/`. El directorio
  `pytest-cache-files-javier-stream/` tiene acceso denegado; no se modifica.
- Histórico reproducido: Macro-F1 `0.6194934236214421`, accuracy `0.647`.
  Se conservan implementación, informes, predicciones de referencia y submissions.
  Los modelos binarios de la reproducción importada no están presentes localmente.
- Entorno de esta ejecución: Python 3.12.6 y CPU, frente a Python 3.13.7/CUDA
  documentados en la reproducción histórica. Además se hace independiente de IDs
  el orden RNG de corrupción. El delta final no aísla causalmente sólo el efecto
  de la selección histórica: incluye esas diferencias reproducibles de ejecución.
- El bloque final de la solicitud corresponde a esta rama. El experimento Direct
  Robust ya está terminado en `exp/v4-direct-robust-javi`; no se mezcla aquí.

## Auditoría de selección histórica

La primera lectura registrada de VALID fue a las 02:10:04 UTC, tras `5ccda8d`.
`reports/final_protocol.md` reconoce que se rechazaron los biases al observar
su resultado en VALID. `ffc656e` añadió componentes y selección posteriores.
Congelar una segunda receta no devuelve independencia al holdout conocido.
Esta reconstrucción tampoco crea un holdout nuevo: comprueba una nueva selección
exclusivamente TRAIN, condicionada en una arquitectura investigada anteriormente.

La comparación del AST con `5ccda8d` confirma que `ranking.py` es idéntico;
`templates.py` y `augmentation.py` sólo cambian imports sin uso. En `features.py`
sólo se fijó una variable de cierre en `tabular`, que esta receta no utiliza.
Los patrones, MCC, templates, normalización y tasas de corrupción son anteriores
a la primera lectura. No se presupone esa independencia para los módulos compactos
y de contexto de pagos, inexistentes en aquel commit.

| Componente | Clasificación inicial | Tratamiento |
| --- | --- | --- |
| Target, cutoff, ocho clases y métrica | CLEAN | Invariables |
| Agrupación por cliente; exclusión de IDs/targets | CLEAN | Aserciones y sanity tests |
| Normalización, patrones semánticos, MCC, templates | TRAIN_SELECTED | Presentes antes de primera lectura; conservar |
| Streams: tolerancia log 0.035, mínimo 3, top 2, decaimiento 60 días | TRAIN_SELECTED | Presentes antes de primera lectura; conservar |
| Perfiles precio, cuantiles .005/.995 | TRANSDUCTIVE_UNLABELED | Sólo pretrain disjunto; ablation sin perfiles |
| Asignación suave 3×semántica + MCC + .05; corte .4/.02 | TRAIN_SELECTED | Presente antes de primera lectura; comparar asignación dura |
| Tasas de corrupción y confusión MCC | TRANSDUCTIVE_UNLABELED | Hipótesis previas de auditoría de covariables VALID/TEST; congelar, sin nueva inspección |
| Inclusión de augmentations | VALID_INFLUENCED | Volver a comparar con entrenamiento original |
| Compactación, contexto de pagos/reembolsos | UNCERTAIN | Incorporados después; ablation explícita |
| Umbrales compactos de actividad/cadencia 5–120, 1.35 | UNCERTAIN | Ablation sin features derivadas de esos umbrales |
| Matching de pagos/refunds .04, ventana 10 días | UNCERTAIN | Comparar con eliminación completa del contexto |
| Detector especializado none y sustitución jerárquica | UNCERTAIN | Comparar con ranker conjunto de ocho clases |
| Parámetros ranker compacto: 500 árboles, 15 hojas, mínimo 120, L2 10 | UNCERTAIN | Comparación pequeña predeclarada de 7/15 hojas |
| Parámetros legacy LGBM/XGB, softmax unitario | TRAIN_SELECTED | Ya congelados antes del primer holdout; control de normalización simple |
| Semillas 42, 17, 2026; número de modelos | UNCERTAIN | Todas predeclaradas; comparar una/tres; no elegir mejor seed |
| Peso legacy .25; criterio de ganancia .005 en stress | UNCERTAIN | Descartar criterio histórico; comparar pesos 0/.25/.50 por OOF |
| Rechazo de biases y decisión final sin biases | VALID_INFLUENCED | No reutilizar vector histórico; comparar argmax y ajuste pequeño por prior TRAIN |
| Temperatura/normalización/candidate aggregation | UNCERTAIN | Comparar normalización histórica, temperatura 1.5 y logits promediados |
| Mínimo 2, experto auxiliar, variantes rechazadas | UNCERTAIN | No se recuperan: quedan fuera del alcance quirúrgico |

## Protocolo predeclarado antes de ejecutar OOF

1. Cinco folds estratificados sobre una fila por `client_id`, seed 42, igual a la
   baseline seed 42. Todas las vistas del cliente permanecen juntas. IDs ordenados.
2. Desarrollo sólo TRAIN. Perfiles fijos aprendidos del pretrain sin etiquetas
   independiente. Ninguna lectura nueva de covariables VALID/TEST durante selección.
   La única herencia transductiva adicional son las tasas de corrupción ya congeladas.
3. Transformaciones deterministas por cliente, sin targets; vocabulario de categorías
   globales y esquema de features restringidos al training fold. No TF-IDF ni SVD.
4. Tres vistas original/medium/severe con las tasas históricas. El RNG de corrupción
   usa orden de hashes de historias, sin client_id; preserva rename y row shuffle.
   Las tres vistas tienen igual peso; todos los clientes tienen el mismo peso total.
5. Métrica primaria de selección: Macro-F1 OOF **original**, ocho clases exactas,
   zero_division=0. Accuracy, F1/precision/recall por clase y confusion matrix.
   Stress OOF se reporta, pero no cambia el criterio. Nada se selecciona por VALID.
6. Catálogo pequeño cerrado: full; sin perfiles; sin augmentation; una seed;
   principal ranker; sin detector none; asignación dura; logits simples;
   sin contexto de pagos; sin umbrales de cadencia; siete hojas;
   sin legacy; legacy .50; temperatura 1.5; bias prior TRAIN (coeficiente .25).
   No combinaciones cruzadas ni ampliaciones tras resultados. Empates exactos:
   menor número de estimadores, luego orden del catálogo. Todas las negativas se guardan.
7. Estabilidad del ganador con seeds de partición 42/17/2026 (sin elegir la mejor),
   más desglose de las tres semillas de modelo del ensemble.
8. Permutación target seed 20260925 y tres controles de azar; rename de IDs,
   shuffle de filas, target poisoning, cutoff y solapamientos. La permutación
   reentrena la receta elegida sin volver a seleccionar variantes.
   Un F1 de permutación superior a .20 bloquea la congelación y exige investigación.
9. Congelar configuración completa, features, parámetros, seeds, pesos, reglas,
   perfiles, hashes de código y datos, y probabilidades OOF antes de VALID.
10. Entrenar sólo TRAIN. Persistir probabilidades VALID/TEST y hashes antes de
    abrir labels VALID; evaluación final única protegida por recibo exclusivo.
    Submission separada si Macro-F1 final >= .52; condición fijada ahora.

La comparación de ablaciones comparte folds. La selección y el score OOF del
ganador reutilizan TRAIN: es evidencia de desarrollo, no estimación anidada sin
sesgo de selección. Las pruebas de estabilidad tampoco eliminan esa limitación.

## Ejecución y reproducción

Runner aislado: `scripts/run_stream_identity_clean.py`; implementación nueva:
`src/ubs_recurrence/clean_protocol.py`. La implementación histórica no se modifica.
Las etapas son `prepare`, `develop`, `stability` (incluye permutación e invariancia),
`freeze`, `predict`, `evaluate`, `report`, `verify`.

En esta máquina se usa `.venv/runtime_v2/python.exe -X utf8` como intérprete.
En un entorno nuevo se instala el paquete con sus dependencias de desarrollo y
las versiones documentadas en la configuración final. Ejemplo de llamada:

```text
python -X utf8 scripts/run_stream_identity_clean.py develop
```

Después de la congelación, el desarrollo queda bloqueado. Para una reproducción
en un clon con outputs nuevos se ejecuta `rebuild`, `predict`, `verify`.
`verify` compara hashes de predicciones y no abre labels VALID. No se repite
`evaluate` como parte de esa comprobación.

Las auditorías independientes de estabilidad/permutación/invariancia se ejecutan
con dos workers que comparten features inmutables; cada estimador usa cuatro hilos.
La selección inicial y los quince candidatos no cambian. Antes de mejorar esa
orquestación y los controles de reproducción, se archivaron los 17 archivos exactos
del desarrollo en `outputs/stream_identity_clean/development_sources.zip`, comprobados
contra los hashes de `development/specification.json`. El módulo numérico y el
criterio de selección permanecieron idénticos. Los hashes del código congelado
normalizan finales de línea; los hashes de datos y predicciones son byte a byte.

Los hashes de TRAIN labels, TRAIN transactions y pretrain coinciden con el
manifiesto histórico. Los de VALID/TEST se comprueban en la etapa final, después
de cerrar el desarrollo. El hash de labels VALID se comprueba después de guardar
predicciones y crear el recibo exclusivo de acceso.

## Selección cerrada antes de VALID

Ganador: `legacy_half`, OOF Macro-F1 `0.673810376395381`.
Se conservan tres semillas, augmentations, contexto de pagos, detector de none,
asignación suave y temperatura 1. La mezcla pasa a 50% promedio de los tres
rankers compactos con su detector none y 50% promedio de los tres expertos legacy.
No hay biases de decisión. La receta fue elegida sólo por OOF; no se combinaron
después piezas de otras ablaciones para intentar subir el score.

Con semillas de partición 42/17/2026: mean `0.66838202903315`, std poblacional
`0.0038482054149893883`, min `0.6653319817744021`, max `0.673810376395381`.
Semillas de modelo aisladas: `0.6662157278290333`, `0.6723507567069853`,
`0.6745218521963389`. Ninguna semilla explica por sí sola una ganancia excepcional;
la semilla 2026 aislada supera ligeramente al promedio, pero no se reelige por ello.
Recomponer el ensemble a partir de esos ajustes independientes reproduce todas
las decisiones OOF y sus probabilidades con error máximo `2.22e-16`, en las tres vistas.

Permutación target: Macro-F1 `0.07011612730541604`; referencia mayoritaria
`0.057470157874470545`; controles uniformes empíricos `0.120561`, `0.126827`,
`0.121891`. El modelo permutado tiende a la clase mayoritaria, sin señal útil.
El valor teórico `0.125` corresponde a accuracy uniforme, no a Macro-F1 esperado
con clases desbalanceadas. No se usa ese valor como umbral de éxito.
Rename, shuffle y target poisoning en 24 clientes TRAIN, tres por clase:
error máximo de features y probabilidades `0.0` en las tres vistas.

Todos los clientes tienen tres vistas de entrenamiento con igual peso; se mantiene
la escala de pérdida histórica. Al retirar augmentations también cambia la escala
de regularización relativa al número de filas. La ablation no aísla causalmente
ese efecto de regularización.

Checks antes de congelar: `pytest` completo y ambos controles Ruff aprobados;
el detalle final se registra abajo. Ninguna etiqueta VALID se ha abierto en esta
reconstrucción hasta este punto.

## Congelación antes de VALID

FROZEN_BEFORE_VALID: YES

Config: `configs/stream_identity_clean_frozen.json`

SHA256: `2464457bbd868ff71b2c05e5f925aefbe36ee52294fc8ce72237184f5d513034`

Base commit: `d5dddfd8d93df61d3a6421f860428bf491c995f7`

## Resultados ejecutados

| Variant | OOF Macro-F1 | Accuracy | Δ vs full | Medium F1 | Severe F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 0.661873024 | 0.691500 | +0.000000 | 0.634408 | 0.612453 |
| no_unlabeled | 0.658363250 | 0.688000 | -0.003510 | 0.632474 | 0.598482 |
| no_augmentations | 0.671134983 | 0.705000 | +0.009262 | 0.346809 | 0.141027 |
| no_multiseed | 0.655213961 | 0.685500 | -0.006659 | 0.634237 | 0.614544 |
| principal_lightgbm | 0.637299464 | 0.665500 | -0.024574 | 0.623584 | 0.606307 |
| no_none_detector | 0.653500551 | 0.681000 | -0.008372 | 0.628693 | 0.607349 |
| no_soft_assignment | 0.657948585 | 0.689000 | -0.003924 | 0.638593 | 0.605819 |
| simple_postprocessing | 0.658867561 | 0.686500 | -0.003005 | 0.628217 | 0.609527 |
| no_payment_context | 0.640262455 | 0.670000 | -0.021611 | 0.605710 | 0.569500 |
| no_cadence_thresholds | 0.664197851 | 0.693000 | +0.002325 | 0.639106 | 0.606644 |
| seven_leaves | 0.646648371 | 0.678500 | -0.015225 | 0.629212 | 0.595516 |
| no_legacy | 0.646228824 | 0.678000 | -0.015644 | 0.626967 | 0.600898 |
| legacy_half | 0.673810376 | 0.701500 | +0.011937 | 0.638174 | 0.609738 |
| temperature_1_5 | 0.661475809 | 0.692500 | -0.000397 | 0.633901 | 0.609054 |
| train_prior_bias | 0.658340883 | 0.685500 | -0.003532 | 0.631396 | 0.619108 |

Ganador TRAIN-only: **legacy_half**. Todas las métricas y OOF por cliente/clase se conservan en `outputs/stream_identity_clean/development/`.

### Estabilidad TRAIN-only

Semillas de partición: `{'42': 0.673810376395381, '17': 0.6660037289296674, '2026': 0.6653319817744021}`.
Mean=0.668382029; std=0.003848205; min=0.665331982; max=0.673810376.
Semillas de modelo con folds fijos: `{'42': 0.6662157278290333, '17': 0.6723507567069853, '2026': 0.6745218521963389}`; std=0.003517321.

### Sanity checks

Permutación target: F1=0.070116127. Controles aleatorios: `{'42': 0.12056127967868663, '17': 0.12682680938864546, '2026': 0.12189060202183177}`.
Rename/shuffle/target poisoning: `{'clients': 24, 'scenarios': ['original', 'medium', 'severe'], 'max_feature_error': 0.0, 'max_probability_error': 0.0, 'probability_errors_by_scenario': {'original': 0.0, 'medium': 0.0, 'severe': 0.0}, 'rename_ids': True, 'row_shuffle': True, 'target_poisoning': True, 'cutoff_checked': True, 'train_pretrain_overlap': False}`.
Cutoff comprobado por lector de datos y transformador; splits TRAIN/VALID/TEST/pretrain disjuntos comprobados antes de la lectura final de labels.

### Evaluación oficial única

| Model | Macro-F1 | Accuracy |
| --- | ---: | ---: |
| V1 histórico | 0.271024266 | 0.266 |
| V2 histórico | 0.391549456 | 0.424 |
| Stream Identity historical | 0.619493424 | 0.647 |
| Stream Identity clean frozen | 0.634819707075 | 0.662000 |

Delta vs V2: +0.243270251075; delta vs histórico: +0.015326283454.

| Class | OOF F1 | VALID precision | VALID recall | VALID F1 | Support | Predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 0.620321 | 0.652632 | 0.696629 | 0.673913 | 89 | 95 |
| gym | 0.649874 | 0.658537 | 0.669421 | 0.663934 | 121 | 123 |
| insurance | 0.717241 | 0.623656 | 0.585859 | 0.604167 | 99 | 93 |
| mobile | 0.683805 | 0.702128 | 0.634615 | 0.666667 | 104 | 94 |
| music | 0.621212 | 0.602410 | 0.537634 | 0.568182 | 93 | 83 |
| software | 0.646766 | 0.602151 | 0.538462 | 0.568528 | 104 | 93 |
| streaming | 0.633708 | 0.600000 | 0.556701 | 0.577540 | 97 | 90 |
| none | 0.817556 | 0.714286 | 0.802048 | 0.755627 | 293 | 329 |

Confusion matrix: filas reales, columnas predichas, orden `cloud, gym, insurance, mobile, music, software, streaming, none`.

| Real / predicted | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 62 | 2 | 3 | 1 | 0 | 4 | 8 | 9 |
| gym | 4 | 81 | 5 | 3 | 3 | 5 | 5 | 15 |
| insurance | 4 | 7 | 58 | 3 | 6 | 4 | 5 | 12 |
| mobile | 2 | 6 | 5 | 66 | 3 | 4 | 3 | 15 |
| music | 4 | 6 | 8 | 5 | 50 | 6 | 3 | 11 |
| software | 5 | 7 | 3 | 4 | 6 | 56 | 5 | 18 |
| streaming | 6 | 5 | 1 | 4 | 6 | 7 | 54 | 14 |
| none | 8 | 9 | 10 | 8 | 9 | 7 | 7 | 235 |

### Calibración y recursos

| Split | Log loss | Brier | Entropy | Confidence |
| --- | ---: | ---: | ---: | ---: |
| OOF original | 0.926500 | 0.436327 | 1.263203 | 0.581837 |
| VALID | 1.102484 | 0.503395 | 1.310047 | 0.555239 |

Preparación: 637.3s. Catálogo OOF: 1835.5s; pico RSS 1781.9 MiB. Ajuste final e inferencia: 327.0s; pico RSS 2858.2 MiB.
Features del fold 0 elegido: 221. CPU; entorno exacto en config. No se estiman comparaciones de tiempo con GPU histórica.

### Conclusión metodológica

Clasificación orientativa: **ROBUST_BREAKTHROUGH**. La selección de esta ejecución usa sólo TRAIN y no realiza un segundo ajuste tras VALID. El score histórico sigue siendo reproducible; esta reconstrucción no convierte el VALID previamente observado en un holdout nuevo. La arquitectura y las hipótesis de shift heredadas siguen condicionando el experimento. No se encontró inclusión predictiva de client_id, target o transacciones futuras; sí existía contaminación histórica por selección sobre VALID.

Submission: `outputs/predictions/submission_stream_identity_clean.csv`. Validación: YES; sin sobrescribir entregas anteriores.

Probabilidades OOF original/medium/severe del ganador y métricas completas quedan disponibles para stacking en `reports/stream_identity_clean_oof_{original,medium,severe}.csv`. El JSON resumido versionado está en `reports/stream_identity_clean_summary.json`.

### Verificaci?n final y trazabilidad

| Check | Resultado |
| --- | --- |
| `python -m pytest -q` | PASS: 41 tests |
| `python -m ruff check .` | PASS |
| `python -m ruff format --check .` | PASS: 64 archivos |
| Pipeline: hashes de configuraci?n/c?digo/predicciones | PASS, sin segunda lectura de labels |
| Validador independiente de submission | PASS: 1.000 IDs ?nicos; 0 faltantes, extra, duplicados, nulls o clases inv?lidas |
| Artefactos anteriores | Todos los hashes iniciales conservados |
| OOF para stacking | 2.000 clientes; ocho probabilidades en orden oficial; original/medium/severe |

Cronolog?a UTC: configuraci?n congelada **10:40:34**; predicciones y recibo
persistidos **10:46:20**; ?nica apertura de labels **10:48:27**.
La configuraci?n mantiene SHA256
`2464457bbd868ff71b2c05e5f925aefbe36ee52294fc8ce72237184f5d513034`.
La submission tiene SHA256
`9542068c0b2df50e58861a1c6bb702e9820d51dc2192d06eaf14b3dfe8f4c978`.

El mayor aumento de F1 por clase frente al hist?rico aparece en gym
(aproximadamente +0.0556); none sube de 0.7484 a 0.7556.
Insurance y music bajan ligeramente (aproximadamente -0.0012 y -0.0017).
Se conserva el resultado completo, sin promover variantes adicionales.
ROBUST_BREAKTHROUGH es una etiqueta interna orientativa, no un criterio oficial UBS.

### Resultado solicitado

STREAM_IDENTITY_HISTORICAL_VALID_F1: 0.6194934236214421

STREAM_IDENTITY_CLEAN_OOF_F1: 0.673810376395381

STREAM_IDENTITY_CLEAN_VALID_F1: 0.6348197070753525

DELTA_CLEAN_VS_V2: 0.2432702510753525

DELTA_CLEAN_VS_HISTORICAL: 0.015326283453910405

FROZEN_BEFORE_VALID: YES

ROBUSTNESS_STATUS: ROBUST_BREAKTHROUGH

SUBMISSION_VALID: YES
