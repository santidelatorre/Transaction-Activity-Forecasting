# Integración de infraestructura alrededor de Stream Identity

Fecha: 2026-09-25
Rama: `integration/stream-identity-main`
Base predictiva: `research/import-v2-stream-identity`
Infraestructura auditada: `origin/main` en `051ce64a7cf0ab999f8aacb81fa405d5fa0257cf`

Esta integración fue semántica. No se hizo merge, cherry-pick ni cambio de rama;
los archivos de `origin/main` se inspeccionaron con `git show` y `git ls-tree`.
Las dos ramas no tienen merge-base, por lo que un merge automático tampoco
habría podido preservar una historia común fiable.

## Decisiones por componente

| Componente | main antiguo | Stream Identity | Acción |
| --- | --- | --- | --- |
| Pipeline predictivo | V1/V2/V3 y smoke genérico | Stream Identity limpio congelado | `KEEP_STREAM` |
| Config y fuentes con hash | configs TOML legacy | freeze JSON y `ubs_recurrence` | `KEEP_STREAM` |
| GitHub Actions | pytest y Ruff en PR | ausente | `ADAPT` |
| pre-commit | Ruff, YAML y archivos grandes | Ruff y pytest locales | `MERGE` |
| pyproject/dependencias | bounds y tooling amplio | dependencias reales del modelo | `MERGE` |
| Validator y contrato UBS | validator sólido con default V2 | núcleo oficial ya portado | `ADAPT` |
| Tests genéricos | CSV, métrica, imports y CLI | leakage, freeze y determinismo | `MERGE` |
| Quality gate | acoplado a V1 y `origin/main` | receipts y hashes propios | `DROP_OBSOLETE` |
| Runners/modelos legacy | V1/V2/V3 | runner Stream Identity | `DROP_OBSOLETE` |
| Documentación | contrato útil e historia legacy | protocolo limpio completo | `ADAPT` |
| Outputs y modelos | métricas/submissions antiguas | submission limpia aprobada | `DROP_OBSOLETE` |

## 1. Recuperado de main

- Workflow de CI con `pytest`, `ruff check .` y `ruff format --check .`.
- Hooks de YAML, conflictos, whitespace, fin de fichero y archivos grandes.
- Límites de Python y dependencias, configuración explícita de pytest/Ruff y
  extras de desarrollo.
- Contrato oficial UBS, setup para PowerShell/Bash, guía de contribución y
  template de pull request.
- Casos históricos independientes del modelo para métrica oficial, ocho clases,
  cutoff, columnas, IDs, duplicados, orden del sample, CLI e imports.

## 2. Descartado

- Runners y tests específicos de V1, V2 y V3.
- Pipeline smoke/genérico y sus configs, que representan otro objetivo.
- Quality gate V1, thresholds contra scores legacy y ejecución opcional de V1.
- Logger SQLite de experimentos, duplicado por los ledgers y receipts actuales.
- Documentos de arquitectura centrados en `transaction_forecasting`, VS Code,
  notebooks, métricas, submissions y artefactos legacy.
- Modelos binarios, datasets y outputs grandes.

## 3. Adaptado

- `scripts/validate_submission.py` usa por defecto la submission limpia actual.
- `scripts/reproduce_stream_identity_submission.py` permite reproducir desde la
  rama de integración reutilizando el runner congelado sin modificar su hash.
  Exige un workdir vacío, valida el CSV y compara bytes con la referencia.
- README y contrato oficial presentan Stream Identity como baseline principal y
  separan tests rápidos de training/benchmark manual.
- Ruff conserva el ancho histórico de 88. Se ignora únicamente `E741` para no
  renombrar variables dentro de fuentes predictivos congelados.
- Los hooks usan el entorno del proyecto para que la versión de Ruff sea la misma
  localmente y en CI.

## 4. Tests añadidos

- `tests/test_official.py`: métrica de ocho clases, alineación por ID, clases
  ausentes, esquema, duplicados, cutoff, orden, line endings y CLI.
- `tests/test_cli.py`: `--help` del evaluador, validator, runner congelado y
  lanzador de reproducción.
- `tests/test_import.py`: imports del paquete y consistencia del orden de clases.

Se conservaron los tests Stream Identity ya existentes de leakage, folds por
cliente, cutoff, target poisoning, invariancia, determinismo, freeze, hashes,
reanudación y evaluación única de VALID.

## 5. CI integrado

`.github/workflows/ci.yml` usa Python 3.12, instala `.[dev]` y ejecuta los tres
checks requeridos. No abre VALID ni ejecuta fitting completo. Los benchmarks y
la regeneración de la submission son acciones manuales.

## 6. Dependencias cambiadas

- Se preservaron NumPy, pandas, SciPy, scikit-learn, PyArrow, CatBoost,
  LightGBM, XGBoost y joblib.
- Se añadieron bounds compatibles con Python 3.11–3.13 y el entorno congelado.
- Se declararon `pre-commit`, `psutil`, pytest y Ruff como herramientas dev.
- `requirements-lock.txt` incorpora `psutil==7.2.2`, `ruff==0.16.8` y
  `pre-commit==4.6.2`; las versiones predictivas exactas no cambiaron.
- Una instalación editable limpia en Python 3.12.10 completó correctamente.

## 7. Receta predictiva

No cambió. Ningún archivo listado en `source_sha256`, ni
`configs/stream_identity_clean_frozen.json`, ni la submission aprobada fue
modificado.

- Envelope interno: `2464457bbd868ff71b2c05e5f925aefbe36ee52294fc8ce72237184f5d513034`.
- SHA-256 del JSON: `a0550bf35a053fcaaac129474c17cd1246def1751e39ffe36c51365c173293b8`.
- SHA-256 de la submission: `9542068c0b2df50e58861a1c6bb702e9820d51dc2192d06eaf14b3dfe8f4c978`.
- Receta: `legacy_half`; OOF congelado: `0.673810376395381`.

## 8. Resultado de pytest

`58 passed` en Python 3.12.10 con pytest 8.4.2. Duración: 106.04 s. Hubo
un warning local de escritura de `.pytest_cache` por permisos de OneDrive, sin
fallos ni impacto en resultados.

## 9. Resultado de Ruff

- `ruff check .`: PASS.
- `ruff format --check .`: PASS, 71 archivos ya formateados.
- Versión: Ruff 0.16.8.

## 10. Resultado del validator

PASS sobre `outputs/predictions/submission_stream_identity_clean.csv`:

- 1.000 filas y 1.000 IDs únicos.
- 0 IDs ausentes, extra o duplicados.
- 0 etiquetas inválidas, nulas o vacías.
- Esquema y alineación con el sample: PASS.

## 11. Riesgos pendientes

- El training completo no se repitió durante esta integración porque es una
  tarea manual pesada; se verificaron el freeze, todos sus hashes de fuente y
  los CLIs. La reproducción exige extraer primero el fichero de pretraining con
  `scripts/prepare_data.py`.
- GitHub Actions se ha validado mediante sus comandos locales equivalentes; la
  primera ejecución remota ocurrirá tras el push.
- El score VALID no se volvió a consultar y no se tomó ninguna decisión nueva
  con VALID.

STREAM_IDENTITY_RECIPE_CHANGED: NO

PYTEST_PASS: YES

RUFF_PASS: YES

SUBMISSION_VALID: YES

READY_TO_REPLACE_MAIN: YES
