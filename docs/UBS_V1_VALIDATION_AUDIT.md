# Auditoría UBS V1 — revisión de main y Milestone 2

## Revisión actual del 24 de septiembre de 2026

Main revisado: `0199a8b`, merge del
[PR #6 de Santi](https://github.com/santidelatorre/Transaction-Activity-Forecasting/pull/6)
el 24-09-2026 a las 14:33:21 CEST. Contiene `6fbd3f1` (evaluación/reajuste/entrega),
`b9051f1` (integración previa) y `f98e90f` (formato). La auditoría original partía
de `cca95c57ab84b644df987d06bc869a356ec5cc99`; la referencia remota anterior era
`f5e5b37`. `CONTRIBUTING.md`, loader, features, configuración V1 y núcleo oficial
de métricas no cambiaron en main entre ambas revisiones.

### Protección del trabajo y diferencias de Git

Se releyeron `AGENTS.md` y `CONTRIBUTING.md`, se inspeccionaron status y diff,
y se copiaron los **ocho archivos de la auditoría inicial** a
`.pytest_tmp/audit_backup_20260924_143948/`, con SHA-256 por archivo,
`tracked.patch` y status. Se verificó que las copias coinciden byte a byte.
Los ocho son README, runner, `evaluation/official.py`, `ubs/data.py`,
`ubs/features.py`, `ubs/models.py`, este informe y `tests/test_ubs_leakage.py`.
No se copió, ocultó ni cambió `scripts/emergency_submission.py`.

`git fetch origin --prune` solo actualizó referencias. La rama sigue siendo
`dev/carles`, HEAD `cca95c5`, con **1 commit propio y 5 de main pendientes de
incorporar** respecto a `0199a8b`. Los cambios siguientes son una adaptación del
árbol de trabajo; no equivalen a incorporar la historia de main. No se ha hecho
commit, stash, merge, push ni PR. La rama local main permanece en `9f2c8da`.

| Novedad de main | Contraste con el trabajo local | Decisión aplicada |
| --- | --- | --- |
| `ubs.evaluation` alinea Series por ID y rechaza duplicados/mismatch | Complementa el control de IDs en las APIs de modelos | Reutilizado íntegramente; los arrays siguen siendo posicionales por compatibilidad |
| `indexed_prediction` en el runner | Evita perder IDs al puntuar candidatos | Reutilizado en selección interna y puntuación oficial |
| Tiempos, número de features/clientes, `run_config.json`, distribución de entrega | No contradicen la auditoría | Conservados, indicando a qué partición corresponde cada score |
| Refit final con train+valid | Contradice la reserva de valid solicitada y vuelve a hacer encoding de la propia etiqueta | Adaptado a todo train solamente, con encoding OOF; receta congelada antes de evaluar valid |
| Tests de métrica manual y sklearn | Añaden cobertura del nuevo camino con Series indexadas | Conservados, sin crear otro scorer |
| `.gitignore` y exclusiones Ruff de temporales | Compatibles | Incorporadas al árbol de trabajo tal como están en main |
| Casts numéricos en desempate de la heurística | La auditoría ya eliminaba la búsqueda redundante de temperatura | Se mantiene la implementación auditada con métrica compartida y temperatura 1 |
| README anuncia refit train+valid | Contradice el nuevo protocolo | Actualizado; se conservan referencias a la auditoría y a los artefactos de Santi |
| AGENTS de main está orientado a Santiago | Conflicto documental con la política general de la rama y la instrucción del usuario | Se conserva AGENTS de dev/carles; no se cambia de rama |

El [PR #5 de Javi](https://github.com/santidelatorre/Transaction-Activity-Forecasting/pull/5),
abierto al revisar, tiene como diferencia efectiva solo `docs/EDA_RECURRENCE.md`
y `scripts/analyze_javi_recurrence.py` en `9e5e35a`. Los commits históricos sobre
features V2/comparadores/submission fueron retirados del alcance de ese PR;
no se reintroducen desde commits antiguos. Se revisó la EDA sin copiar otro
loader o analizador al runner. `dev/jaime` (`761509a`) añade dos scripts de
preparación/formato; sus riesgos se detallan debajo y no se incorporan.
La consulta de ramas se limitó a commits/rutas del pipeline y documentación.
El cambio de modo de `setup.sh` en Esteban no afecta a esta auditoría.

### Estado actual de los nueve hallazgos originales

Las evidencias históricas, consecuencias y soluciones de cada hallazgo se
conservan en el registro inicial más abajo.

| ID | Severidad | Estado en main `0199a8b` | Solución local actual |
| --- | --- | --- | --- |
| A01 | Alta | Sigue vigente; requiere adaptar la solución al nuevo refit | OOF por cliente dentro del fit interno y también en el reajuste con train completo; nunca con valid |
| A02 | Alta | Sigue vigente: selección/calibración/alpha sobre valid | Selección trasladada al holdout estratificado interno. Valid solo puntúa el modelo congelado. La exposición histórica sigue siendo un riesgo |
| A03 | Media | Sigue vigente; main no cambió los puntos de entrada temporales | Se conservan los controles centralizados de historial en loader y features |
| A04 | Media | Sigue vigente; main no validaba los parámetros ignorados | Se conservan los rechazos de cutoff/target/ventanas incompatibles; se añade fracción interna configurable |
| A05 | Media | Sigue vigente; main conserva el retorno sin columnas al no haber streams | Se mantiene el esquema fijo y evidencia cero, sin fallback a `none` por error |
| A06 | Media | Sigue vigente; main conserva agregados sin unidad | Importes/fees por moneda y streams por moneda/dirección; no se reintroducen magnitudes mezcladas |
| A07 | Media | Corregido parcialmente en main para el scorer; no en las APIs de modelos | Se adopta su scorer indexado y se conservan los controles/alineación de documentos y targets de la auditoría |
| A08 | Media | Sigue vigente; sample tardío e inferencia de IDs sin cambios | Se reutiliza el validador temprano, lectura literal de IDs y validación del CSV releído |
| A09 | Media | Sigue vigente; los casts de main no solucionan clases/columnas ausentes | Se conservan probabilidades de ocho clases, pesos observados, lift cero para clases ausentes y columnas vacías del imputador |

Ningún hallazgo completo ha quedado resuelto exclusivamente por main. A07 sí
tiene una corrección complementaria que se reutiliza, sin duplicarla.

### Riesgos adicionales detectados al revisar las novedades

1. **N01, alta para este protocolo — refit train+valid.** Evidencia:
   `origin/main:scripts/run_ubs_baseline.py:428` concatena ambas particiones y
   ajusta el mapping con sus etiquetas. El score guardado se calculó antes de
   ese refit: no es un score del artefacto final. Esto no prueba leakage desde
   test ni invalida retroactivamente aquel cálculo, pero incumple la reserva de
   valid solicitada y cambia la representación usada para entregar. Solución
   aplicada: `scripts/run_ubs_baseline.py:512` reajusta solo con train, congela
   receta en `:489` y puntúa exactamente ese modelo en valid en `:648`.
2. **N02, alta — formatter de Jaime oculta errores como `none`.** Evidencia:
   `origin/dev/jaime:scripts/format_submission.py:49` y `:55`, `format_predictions`, usa
   `drop_duplicates(keep="first")`, left join y reemplazo de etiquetas inválidas
   o ausentes por `none`. Consecuencia: una entrega formalmente válida puede
   esconder fallos de cobertura o inferencia. No está en main ni se incorpora.
   Recomendación: delegar en `evaluation.official.validate_submission`, que debe
   fallar ante esos casos, y mantener `none` como clase semántica.
3. **N03, media — preparador alternativo sin control temporal completo.**
   Evidencia: `origin/dev/jaime:scripts/prepare_ubs_data.py:52`, `:69` y `:81`,
   `load_training_data`,
   convierte fechas pero no rechaza eventos desde el cutoff ni comprueba la
   disjunción de particiones; devuelve una tabla con target por transacción.
   Consecuencia: no es un sustituto seguro del loader UBS y esa tabla requiere
   excluir explícitamente targets antes de modelar. No se incorpora. Usar
   `ubs.data` para el flujo supervisado y los controles compartidos.
4. **N04, media — dos eventos no prueban regularidad.** Evidencia:
   `ubs/features.py:259` cuenta `interval_std_days <= 3` incluso con un único
   intervalo; la EDA de Javi explica el mismo problema y advierte que `none`
   también tiene pagos regulares. Consecuencia: evidencia de recurrencia
   estadísticamente débil, sin implicar leakage temporal. Riesgo documentado,
   no se cambia esta feature después de ver valid. Recomendación posterior:
   medir soporte por stream y comparar una definición con al menos tres eventos
   exclusivamente en selección interna, con presupuesto de búsqueda predefinido.

### Protocolo experimental aplicado

1. Loader UBS compartido verifica train/valid/test, cutoff, cobertura y sample.
2. `ubs.data.split_training_clients` ordena IDs y crea un holdout estratificado
   de **25 %** con semilla **42**: 1500 clientes de fit, 500 de selección.
   Todas las transacciones del cliente permanecen juntas. El split se registra
   en `selection_protocol.json`; no se particionan filas de transacciones.
3. Categorías, mappings, vocabulario TF-IDF, IDF, imputación, escalado y modelos
   de cada candidato se ajustan solo en los 1500 clientes de fit. El lift de
   cada fila de entrenamiento excluye el fold del cliente mediante cinco folds
   adicionales. Los folds de encoding no dependen del target.
4. Los 500 clientes de selección eligen sesgo de `none`, seis configuraciones
   logísticas, dos CatBoost, el mejor componente ML y diez alphas del ensemble.
   La temperatura permanece en 1; se registran los 19 sesgos y diez alphas.
   Se elige por macro-F1, accuracy y orden fijo en caso de empate. Ese score es
   **internal-selection Macro-F1**, sesgado por selección, no un holdout final.
5. `selected_model.json` congela la receta antes de usar valid o test para
   inferencia. Se reajusta esa receta sobre los 2000 clientes de train, con
   preprocessing aprendido de nuevo solo allí y lift OOF en las filas de train.
   No se incorporan etiquetas ni vocabulario de valid/test al ajuste.
6. El mismo modelo final genera test por ID y se evalúa una sola vez en los
   1000 clientes de valid. El score se publica como **official-validation
   Macro-F1** del modelo congelado, con `official_validation_independent=false`:
   valid ya se utilizó para decisiones en ejecuciones anteriores, incluida la
   de 0.25677. No existe un score históricamente independiente recuperable aquí.
7. El CSV sigue pasando por `ubs.data.validate_submission` y por el núcleo
   oficial; se relee tras escribirlo. Los placeholders del sample nunca son
   targets. No se envía ninguna submission.

La discrepancia comunicada por el usuario entre 0.2463 local y 0.1125 hidden
en Milestone 1 justifica reforzar el protocolo; no identifica por sí sola su
causa. No se conocen las etiquetas ocultas ni se ha demostrado que el cambio
de distribución, el ensemble, el encoding o un error de submission expliquen
esa caída. El leaderboard comunicado no se usa para ajustar hiperparámetros.

### Comprobaciones actuales

La suite completa adaptada pasa **60 tests**: conserva los 57 de la auditoría,
incorpora los dos tests de Santi y añade uno de estratificación/disjunción. El
test del runner se amplía: cambiar todas las etiquetas oficiales y todos los
placeholders no cambia receta ni submission; todos los fits se restringen a
los IDs internos de fit antes de congelar y a train después. Se verifica una
única evaluación oficial posterior a la congelación.

Ruff (`ruff check .`), formato (42 archivos), imports de UBS y runner,
`git diff --check`, y los hooks pre-commit pasan. Los hooks se ejecutaron sobre
todos los archivos versionados y explícitamente sobre los dos archivos nuevos,
sin staging. La caché requiere ejecución autorizada fuera del sandbox.
La última actualización de referencias volvió a confirmar `0199a8b` como main;
Santi, Javi y Jaime permanecían en los commits revisados.

### Ejecución real y resultados separados

Comando ejecutado con el paquete en `PYTHONPATH=src`:

```powershell
python scripts/run_ubs_baseline.py --config .pytest_tmp/ubs_v1_m2_internal_20260924.toml
```

La configuración temporal conserva todos los parámetros de `configs/ubs_v1.toml`
e introduce únicamente rutas nuevas de artefactos. No se cambiaron candidatos,
seed, alpha ni thresholds después de puntuar valid. Duración: **289,77 s**.

| Evaluación | Clientes | Macro-F1 | Accuracy | Interpretación |
| --- | ---: | ---: | ---: | --- |
| Selección interna | 500 | **0.4657876176** | 0.494 | Mejor resultado de una búsqueda; optimista por selección |
| Validation oficial del modelo congelado | 1000 | **0.1911265363** | 0.289 | No usada para seleccionar ni reajustar en este run; históricamente expuesta |
| Test | 1000 | No disponible | No disponible | No hay etiquetas; solo generación y validación del CSV |

**No existe una métrica de validation realmente independiente en esta sesión.**
El resultado anterior **0.25677** sigue siendo un resultado de selección sobre
valid; no se reetiqueta ni se compara como si fuera un benchmark intacto.

Receta congelada: **90 % CatBoost balanceado + 10 % heurística**, seed 42,
CatBoost 350 iteraciones, depth 6, learning_rate 0.05; heurística `none_bias=1.0`,
temperatura 1. La mejora del ensemble sobre CatBoost balanceado en el holdout
interno fue **0.0095591406**, sin afirmación de significación estadística.

| Candidato | Macro-F1 interno |
| --- | ---: |
| Ensemble CatBoost balanceado + heurística | 0.465788 |
| CatBoost balanceado | 0.456228 |
| CatBoost normal | 0.441028 |
| Heurística de recurrencia | 0.438604 |
| Logistic C=0.3, balanced | 0.390054 |
| Logistic C=0.3, sin ponderación | 0.389983 |
| Logistic C=1, balanced | 0.380508 |
| Logistic C=1, sin ponderación | 0.373147 |
| Logistic C=3, sin ponderación | 0.372940 |
| Logistic C=3, balanced | 0.372464 |
| Dummy de mayoría | 0.057396 |

Todos los scores de esa tabla usan los mismos 500 clientes internos. Solo el
candidato seleccionado se evaluó en valid; no hay una búsqueda encubierta entre
modelos usando sus scores oficiales.

| Clase | Soporte interno | Pred. interno | F1 interno | Soporte oficial | Pred. oficial | F1 oficial | Pred. test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 48 | 59 | 0.485981 | 89 | 21 | 0.218182 | 5 |
| gym | 47 | 50 | 0.412371 | 121 | 34 | 0.129032 | 23 |
| insurance | 53 | 83 | 0.470588 | 99 | 46 | 0.220690 | 29 |
| mobile | 48 | 71 | 0.537815 | 104 | 38 | 0.225352 | 33 |
| music | 50 | 40 | 0.355556 | 93 | 18 | 0.072072 | 12 |
| software | 49 | 51 | 0.480000 | 104 | 40 | 0.166667 | 45 |
| streaming | 56 | 51 | 0.336449 | 97 | 23 | 0.100000 | 10 |
| none | 149 | 95 | 0.647541 | 293 | 780 | 0.397018 | 843 |

La brecha entre selección interna y valid es **0.2746610814**. El modelo predice
`none` para el **78 % de valid** aunque su soporte real es **29,3 %**, con
567 falsos positivos en esa clase; en test predice **84,3 % none**. Es una señal
de generalización débil y sobrepredicción de `none`, no un error formal del CSV.
El holdout aleatorio dentro de train no ha demostrado representar la distribución
oficial. No se atribuye la causa exacta a un mecanismo sin nuevas pruebas.

### CSV candidato y recomendación para Milestone 2

Candidato: `outputs/predictions/submission_v1_m2_internal_20260924.csv`.
SHA-256: `994ef9b3838dc36514346ee9367bc1e6d54f80db70e060337ab9ccc1cbf4ee7f`.
Contiene **1000 filas**, dos columnas oficiales, todos y solo los IDs del sample,
orden exacto, IDs únicos y ocho clases permitidas. Validado por el runner tras
relectura y por `evaluation.official validate`. `score_predictions` reproduce
exactamente las métricas oficiales a partir del CSV de valid con filas invertidas.
La recarga de datos vuelve a comprobar cero clientes compartidos y cero eventos
en/después del cutoff en las tres particiones (147459/73898/75761 transacciones).

**Recomendación:** si Milestone 2 debe usar una elección reproducible bajo el
protocolo solicitado, el candidato es el ensemble congelado anterior, porque
ganó la selección interna. **No recomendaría presentarlo como mejora demostrada
ni enviarlo automáticamente:** su generalización externa es débil y la
concentración en `none` requiere revisión humana antes de entregar. No se cambia
a la heurística de 0.25677 ni se retoca alpha usando valid después de ver esta
caída; eso volvería a convertir valid en selección. Para seguir investigando,
predefinir un presupuesto dentro de train y un diseño que contraste estabilidad
entre grupos de clientes, incluyendo soporte de recurrencia y drift. Una
evaluación realmente nueva requeriría otro conjunto intacto.

Deadline comunicado: **antes de las 17:00 CEST del 24-09-2026**. No se ha enviado
el candidato. La revisión de ingeniería está completa; persiste el riesgo de
calidad predictiva y no se garantiza mejorar el hidden score de Milestone 1.

### Archivos y acciones Git pendientes

Cambios locales exactos respecto a HEAD (12 modificados y 2 nuevos):

```text
.gitignore
README.md
configs/ubs_v1.toml
docs/OFFICIAL_CHALLENGE.md
docs/UBS_V1_VALIDATION_AUDIT.md                 (nuevo)
pyproject.toml
scripts/run_ubs_baseline.py
src/transaction_forecasting/evaluation/official.py
src/transaction_forecasting/ubs/data.py
src/transaction_forecasting/ubs/evaluation.py
src/transaction_forecasting/ubs/features.py
src/transaction_forecasting/ubs/models.py
tests/test_ubs_leakage.py                     (nuevo)
tests/test_ubs_v1.py
```

`.gitignore`, `pyproject.toml`, `ubs/evaluation.py` y `tests/test_ubs_v1.py` son
copias de los cambios revisados de main; no son nuevas implementaciones de esta
auditoría. Se conserva el AGENTS general de dev/carles, que difiere del de main
pero no fue modificado en este trabajo. `scripts/emergency_submission.py`
permanece no rastreado, intacto y excluido de la aportación. Datasets, backups,
configuración temporal y outputs están ignorados; no se prepararon para Git.

Falta autorizar un **commit selectivo de estos 14 archivos**, después incorporar
la historia de main en dev/carles y resolver conscientemente los solapamientos
ya identificados; posteriormente push y PR. No hace falta stash para conservar
el estado actual. La copia de seguridad sigue disponible, el índice permanece
vacío y no se ha alterado main.

---

## Registro inicial conservado — histórico anterior a la revisión de main

Las conclusiones, cifras y estado de Git de esta sección corresponden a la
auditoría inicial, no al protocolo actual descrito arriba.

Fecha: 24 de septiembre de 2026. Rama: `dev/carles`.
Base auditada: `cca95c57ab84b644df987d06bc869a356ec5cc99`.
Las líneas de evidencia siguientes corresponden a esa base, para que los
hallazgos sigan siendo localizables después de aplicar las correcciones.

## Conclusión y alcance

El split oficial del ZIP local es disjunto por cliente y todos sus timestamps
están antes del cutoff. El runner ajustaba vocabulario, categorías, imputadores,
escalado y modelos con train; no se encontró uso de etiquetas de test ni de
etiquetas de valid en esas transformaciones. Sin embargo, las asociaciones
supervisadas descripción–familia de cada fila de train incluían su propia
etiqueta. Se corrige mediante cross-fitting por cliente.

El mejor macro-F1 de valid es un **resultado de selección**, porque ese mismo
conjunto decide calibración, modelo y peso del ensemble. No constituye una
estimación independiente ni demuestra una mejora generalizable del ensemble.
La corrección deja este límite explícito y registra todos los intentos de
calibración y mezcla; no convierte el valid ya utilizado en un holdout nuevo.

Se inspeccionaron `ubs/data.py`, `ubs/features.py`, `ubs/models.py`,
`ubs/evaluation.py`, `evaluation/official.py`, el runner, su configuración,
los tests existentes y el contrato local. El script local no versionado
`scripts/emergency_submission.py` queda fuera del alcance y no se modifica.
No se entrenó con el archivo de pretrain sin etiquetas.

## Evidencia sobre los datos locales

ZIP: `data/raw/dataset.zip`, SHA-256
`1afc95470f4e8641601503172be3e698ef9eaf91528d911a6a01a120911634c6`.
Se verificó con `load_ubs_data`, que reutiliza los controles de UBS y del
evaluador oficial; no se añadió un segundo validador.

| Partición | Transacciones | Clientes | Primer timestamp UTC | Último timestamp UTC | En/después del cutoff | Clientes con varias monedas |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| train | 147459 | 2000 | 2024-11-07 00:07:06 | 2025-12-31 23:57:50 | 0 | 1970 |
| valid | 73898 | 1000 | 2024-11-07 00:09:28 | 2025-12-31 23:55:23 | 0 | 991 |
| test | 75761 | 1000 | 2024-11-07 00:33:43 | 2025-12-31 23:58:30 | 0 | 988 |

Intersecciones train/valid, train/test y valid/test: **0, 0 y 0**.
Labels y transacciones cubren exactamente los mismos clientes en train y valid.
Sample: 1000 clientes únicos, iguales a test. Todas las etiquetas tienen
`cutoff_date=2026-01-01`. Monedas observadas: CHF, EUR, USD y GBP.

La política local sigue siendo `timestamp < 2026-01-01T00:00:00Z`.
La inclusividad exacta del corte y el extremo del horizonte de 90 días siguen
pendientes de confirmación del organizador según `OFFICIAL_CHALLENGE.md`.
Los tests prueban el instante límite, fracciones de segundo, offsets UTC y NaT.

## Hallazgos

### A01 — Alta — La etiqueta propia influía en las features de train — Corregido

- **Archivo/línea:** `scripts/run_ubs_baseline.py:170` y `:171`;
  `src/transaction_forecasting/ubs/features.py:81`.
- **Evidencia:** `fit(train, labels)` aprendía lift supervisado y luego
  `transform(train)` aplicaba esa misma tabla a los clientes que la generaron.
  Reproducción sobre el código original en memoria: cambiar únicamente el
  target del primer cliente de la fixture de regresión cambia sus features
  `family_cloud_description_lift` y `family_cloud_recurrence_score`.
- **Consecuencia:** fuga del target hacia la representación de entrenamiento,
  con señal artificial y distribución diferente a la de inferencia. No implica
  que se hayan filtrado etiquetas de valid a train ni permite cuantificar por
  sí sola un sesgo del score de valid.
- **Solución:** `ClientFeatureBuilder.fit_transform` produce lift out-of-fold
  en cinco folds de clientes completos (menos si hay menos de cinco clientes).
  Los folds se fijan por ID ordenado y semilla, sin depender del target.
  Cada fila excluye todas las etiquetas de su fold. La tabla de todo train se
  conserva exclusivamente para transformar valid/test; las categorías se
  aprenden con train. El runner usa esta nueva entrada para train.
- **Protección:** `test_cross_fitted_features_cannot_see_own_label` demuestra
  invariancia de la fila al cambiar su propia etiqueta y comprueba el estado
  de inferencia. No reutilizar `fit(...).transform(train)` para entrenar modelos
  con estas features supervisadas.

### A02 — Alta — Optimismo por selección repetida en valid — Mitigado, riesgo abierto

- **Archivo/línea:** `scripts/run_ubs_baseline.py:123`, `:209` y `:302`;
  `src/transaction_forecasting/ubs/models.py:48`.
- **Evidencia:** el mismo valid selecciona el sesgo de `none`, seis regresiones
  logísticas, dos CatBoost, el componente ML del ensemble y diez alphas. La
  heurística probaba 19 sesgos por cinco temperaturas (95 evaluaciones), pero
  una temperatura positiva no cambia el argmax y no puede calibrarse con F1.
  El informe mostraba el mejor score y delta sin registrar esa búsqueda completa.
- **Consecuencia:** sesgo de selección y riesgo de sobreajuste a valid; un delta
  positivo del ensemble no prueba mejora en clientes nuevos. Repetir ejecuciones
  y decidir nuevas features mirando errores de valid incrementa ese riesgo.
- **Solución aplicada:** fijar temperatura a 1 (el mismo desempate anterior),
  usar la métrica compartida también en la heurística y guardar los 19 sesgos,
  diez alphas, modelos candidatos, configuración y regla de selección en
  `selection_protocol.json`. Informe y summary identifican el score como
  selección, sin inventar un score independiente.
- **Recomendación pendiente:** congelar candidatos y presupuesto de búsqueda;
  seleccionar hiperparámetros/calibración/ensembles con folds de clientes dentro
  de train, reajustando en cada fold vocabulario, imputación, escalado y lift.
  Medir una única vez en un conjunto realmente intacto. Valid ya visto no vuelve
  a ser independiente por cambiar el código. Conservar un registro entre runs;
  el JSON por ejecución no contabiliza decisiones humanas previas.
- **Protección:** el test del runner exige el registro de intentos y la
  advertencia de interpretación. No se afirma que eso elimine el sobreajuste.

### A03 — Media — Entradas directas a features eludían controles temporales — Corregido

- **Archivo/línea:** `src/transaction_forecasting/ubs/features.py:81`, `:118`;
  helpers `build_client_documents` y `build_recurrence_streams` del mismo archivo.
- **Evidencia:** el loader rechazaba timestamps nulos/futuros, pero `fit` podía
  aprender asociaciones de una transacción futura suministrada directamente.
  `transform` solo comparaba `>= CUTOFF`, comparación que no rechaza NaT. Los
  helpers de texto y recurrencia tampoco aplicaban el control.
- **Consecuencia:** notebooks u otros consumidores podían incorporar información
  futura o historiales de fecha desconocida aun usando componentes UBS.
- **Solución:** centralizar `validate_history` en `ubs.data` y llamarlo desde
  loader, fit, transform, documentos y recurrencia. Normalizar timestamps del
  JSONL a UTC antes de calcular features. Reutilizar el control compartido de IDs.
- **Protección:** regresiones de cutoff con offsets y de todas las entradas
  directas ante NaT y el instante exacto del corte.

### A04 — Media — Configuración temporal declarada pero ignorada — Corregido

- **Archivo/línea:** `scripts/run_ubs_baseline.py:31` y
  `src/transaction_forecasting/ubs/features.py:118`.
- **Evidencia:** `load_settings` aceptaba cualquier `data.cutoff`, `data.target`
  y `features.recent_windows`, pero el cálculo usaba constantes de UBS V1.
- **Consecuencia:** un experimento podía declararse de otro corte/target mientras
  seguía usando el target de enero y las ventanas fijas, sin error explícito.
- **Solución:** rechazar valores incompatibles en el runner antes de cargar datos.
  UBS V1 sigue siendo el contrato de enero; no se construyen targets históricos
  reutilizando sus etiquetas. La configuración usada queda en el protocolo.
- **Protección:** el test del runner cambia el cutoff y exige rechazo temprano.

### A05 — Media — Un split sin recurrencias perdía 99 columnas — Corregido

- **Archivo/línea:** `src/transaction_forecasting/ubs/features.py:214`.
- **Evidencia:** el retorno temprano con `streams.empty` omitía todas las
  columnas de recurrencia/familia. En la fixture original, train devuelve 164
  columnas y la variante con un evento por cliente solo 65.
- **Consecuencia:** errores de esquema/KeyError al predecir o analizar errores;
  el caso sin evidencia no podía pasar por el modelo aprendido. No se debe
  resolver convirtiendo un fallo en la etiqueta `none`.
- **Solución:** generar siempre todas las columnas, con evidencia cero donde no
  existen streams. `none` continúa siendo una clase aprendida, sin fallback
  por excepción ni abstención.
- **Protección:** igualdad exacta de esquema y evidencia cero en un split sin
  descripciones repetidas.

### A06 — Media — Magnitudes monetarias incompatibles — Corregido

- **Archivo/línea:** `src/transaction_forecasting/ubs/features.py:36`, `:133`
  y `:268`.
- **Evidencia:** agregados globales de amount/fee y `family_*_typical_amount`
  mezclaban monedas sin conversión. Un stream agrupaba solo cliente y descripción,
  por lo que también mezclaba monedas y direcciones. En train real hay 1970/2000
  clientes con más de una moneda.
- **Consecuencia:** incumplimiento del contrato y medidas de importe/estabilidad
  sin unidad coherente. Es un defecto semántico real, no evidencia de leakage
  temporal ni de uso de etiquetas futuras.
- **Solución:** importes y fees por moneda; retirar magnitudes globales sin unidad
  y el importe típico de familia sin moneda. Separar streams por cliente,
  descripción, moneda y dirección. Los agregados de CV y regularidad son
  adimensionales; no se introduce una conversión FX implícita.
- **Protección:** misma descripción en dos monedas y dos direcciones conserva
  tres streams y los totales de cada moneda se verifican por separado.

### A07 — Media — APIs de modelos aceptaban emparejamientos por posición — Corregido

- **Archivo/línea:** `src/transaction_forecasting/ubs/models.py:104`, `:114`
  y `:175`.
- **Evidencia:** texto, target y features se entregaban a los estimadores en el
  orden recibido, ignorando sus índices de cliente. El runner sí construía ese
  orden correctamente; el defecto aparece al reutilizar las APIs con Series
  reordenadas o incompletas.
- **Consecuencia:** entrenamiento/predicción con texto o etiquetas de otro cliente
  sin error, y diagnósticos inválidos en consumidores nuevos.
- **Solución:** alinear documentos y targets al índice de features antes de
  fitting, transformación y calibración. Rechazar IDs repetidos y conjuntos
  distintos. Los arrays de salida mantienen el orden del índice de features.
- **Protección:** invertir documentos y targets da los mismos coeficientes y
  probabilidades; omitir un documento provoca error. La evaluación vectorial UBS
  conserva su contrato de vectores alineados; para CSV usar `score_predictions`,
  que ya alinea por ID.

### A08 — Media — Validación tardía del sample y coerción de IDs — Corregido

- **Archivo/línea:** `src/transaction_forecasting/ubs/data.py:42`, `:64`, `:91`
  y `:108`; `scripts/run_ubs_baseline.py:391`.
- **Evidencia:** el loader comprobaba únicamente conjuntos del sample, por lo que
  una fila duplicada pasaba hasta la validación final tras entrenar. JSONL podía
  inferir IDs numéricos; CSV trataba identificadores literales como `NA` como nulos.
  La lectura de etiquetas no compartía todos los controles de IDs del evaluador.
- **Consecuencia:** entrenamiento costoso con un sample inválido, pérdida de
  identidad en archivos alternativos y errores tardíos de cobertura. El ZIP
  auditado no presenta esos IDs ni duplicados; son riesgos reproducibles de API.
- **Solución:** conservar tipos originales en JSONL e IDs literales en CSV;
  compartir `validate_labels`/`validate_client_ids` con `evaluation.official` y
  aplicar el validador UBS al sample antes de entrenar. Los placeholders del
  sample no se interpretan como etiquetas. Validar también la relectura del CSV.
- **Protección:** regresiones de IDs `001`/`NA`, sample duplicado/ID vacío/esquema
  extra, y runner completo con el sample en orden inverso.

### A09 — Media — Clases ausentes y columnas numéricas vacías rompían modelos — Corregido

- **Archivo/línea:** `src/transaction_forecasting/ubs/models.py:21`, `:92` y
  `:158`; `src/transaction_forecasting/ubs/features.py:92`.
- **Evidencia:** el reordenado suponía que `classes_` contenía las ocho clases;
  pesos balanceados exigían todas las clases incluso si faltaban. Los conteos
  de clase del lift usaban NaN para clases ausentes. El imputador descartaba una
  columna totalmente vacía pero `feature_importance()` mantenía su nombre.
- **Consecuencia:** KeyError/ValueError en folds pequeños o diagnóstico; posible
  desalineación de nombres y pesos. No se detecta ausencia de clases en el train
  oficial, pero el cross-fitting exige soportar folds con clases ausentes.
- **Solución:** conteos ausentes y lift de clases no observadas a cero (sin
  inventar evidencia mediante smoothing), pesos solo de clases observadas,
  probabilidades cero en las posiciones de clases no aprendidas, y conservar
  columnas vacías durante la imputación. Macro-F1 sigue promediando ocho clases.
- **Protección:** ambos modelos entrenan con dos clases y producen ocho columnas
  de probabilidades; se comprueba suma uno, ceros de clases ausentes y nombres
  de feature importance. No se prometen modelos supervisados útiles con una
  única clase o un vocabulario completamente vacío.

## Controles correctos conservados

- **Macro-F1:** `evaluation.official.classification_metrics` calcula por clase
  `2*TP/(2*TP+FP+FN)` y divide la suma entre ocho. Un denominador cero aporta cero.
  Los tests existentes verifican a mano `(6 + 2/3)/8` con un falso `none`, y
  `1/8` cuando solo hay una clase perfectamente predicha. No se añade otro
  evaluador; `ubs.evaluation` conserva su interfaz y delega al núcleo compartido.
- **None:** participa con el mismo peso que las siete familias. No se usa para
  rellenar predicciones ausentes ni capturar errores. El sesgo de la heurística
  es un hiperparámetro seleccionado con valid, registrado como tal.
- **Fitting:** TF-IDF, medianas, StandardScaler y CatBoost se ajustan únicamente
  con train. Tests adicionales comprueban que inferir con valores extremos,
  palabras y categorías nuevas no cambia el estado aprendido.
- **Features futuras:** solo entran las columnas históricas permitidas. El ID
  sirve como índice/unión; no entra como predictor. `days_since_last`, cadencia
  y due score usan el corte fijo y eventos previos, no fechas futuras observadas.
  Palabras como nombres de familia en la descripción son texto histórico; su
  presencia no demuestra fuga, aunque pueda ser un atajo del generador sintético.
- **Alineación y entrega:** el runner reindexa labels a las features y construye
  una Series de predicciones indexada por `client_id`; luego hace `map` contra
  el sample. El validador compartido rechaza duplicados, omisiones, extras y
  clases inválidas; UBS exige además el orden exacto del sample. La escritura
  se relee y vuelve a validar. No se envía ninguna submission.

## Verificación y límites de revisión

- Antes de editar: **35 tests pasaban**, Ruff y formato correctos. La primera
  ejecución de pytest tuvo tres errores de permisos temporales; una ejecución
  autorizada confirmó la base. No eran fallos del código.
- Tras las correcciones: **57 tests pasan**, incluidos 22 casos nuevos que
  protegen los riesgos anteriores y ejecutan el runner real con fixtures.
- Se instalaron en `.venv` las dependencias de modelos ya declaradas, ausentes
  en el entorno inicial: scikit-learn 1.9.1 y CatBoost 1.2.10. No se modificó el
  contrato de dependencias ni se versionaron entornos.
- La ejecución con el ZIP local utiliza la configuración V1 y rutas nuevas:
  `outputs/metrics/ubs_v1_audit_20260924/` y
  `outputs/predictions/submission_v1_audit_20260924.csv`, ignoradas por Git.
  Sus scores son selección sobre valid, no una comparación causal contra V1
  anterior ni una estimación sobre test oculto.
- Ejecución real completada: 234 features; mejor candidato
  `recurrence_heuristic`, macro-F1 de selección **0.2567698721**, accuracy **0.251**.
  El ensemble mejora **0.0093169356** respecto a su componente ML en valid,
  pero no supera la heurística y no se selecciona. CSV: **1000 filas válidas**.
  El CLI `evaluation.official validate` confirma la entrega y `score_predictions`
  reproduce exactamente macro-F1 y accuracy del runner con las filas invertidas.
  Este run usa el lift de las ocho clases presentes; la protección adicional
  para clases ausentes no altera sus folds, comprobados con la misma semilla.
- Ruff, formato, `git diff --check` y pre-commit verificados. Se usó una caché
  de hooks nueva e ignorada; sus manifiestos no resultaban accesibles dentro del
  sandbox, por lo que los hooks se verificaron con ejecución autorizada fuera de él.
  Los nuevos archivos sin staging se comprobaron también de forma explícita.
- El cambio de esquema monetario y la codificación OOF requieren reentrenar;
  no reutilizar modelos persistidos con las columnas anteriores.
- `origin/main` se actualizó por fetch. Solo difería del árbol de trabajo base
  en instrucciones de ramas de `AGENTS.md`; no había cambios de pipeline por
  incorporar. No se hizo merge ni se modificó la rama local `main`.
- El archivo local ajeno conserva SHA-256
  `AD525BAEF46EEE1B2E5B31AE2D5D774E680478D7EDE490B44B517B433C9E1E08`.
  No se hizo staging, commit, push, PR ni merge. Datasets y artefactos quedan
  locales; la aportación consiste en código, tests y este informe.
