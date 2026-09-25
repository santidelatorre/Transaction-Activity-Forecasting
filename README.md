# UBS recurring merchant forecasting

La baseline principal es **Stream Identity clean frozen**. Reconstruye streams
recurrentes por cliente y combina rankers de familia con un detector dedicado de
`none`. La selección y las pruebas de estabilidad se hicieron solo con TRAIN.

| Evaluación | Macro-F1 | Uso |
| --- | ---: | --- |
| TRAIN-only OOF | **0.673810376395381** | selección de la receta congelada |
| VALID limpio congelado | **0.634819707075** | una evaluación posterior al freeze |

VALID no debe reutilizarse para elegir features, parámetros, reglas o umbrales.
La receta aprobada está en
[`configs/stream_identity_clean_frozen.json`](configs/stream_identity_clean_frozen.json)
y el protocolo metodológico completo en
[`reports/stream_identity_clean_protocol.md`](reports/stream_identity_clean_protocol.md).
El informe de esta integración está en
[`reports/stream_identity_integration.md`](reports/stream_identity_integration.md).

## Instalación y comprobaciones rápidas

Se admite Python 3.11–3.13; la receta congelada se ejecutó con Python 3.12.6.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
pre-commit run --all-files
```

Los tests cubren el contrato UBS, ocho clases fijas, alineación por `client_id`,
duplicados, nulls, cutoff, aislamiento de folds, leakage, determinismo, hashes,
reanudación y CLIs. No entrenan el modelo completo.

## Submission limpia

El CSV aprobado y versionado es:

```text
outputs/predictions/submission_stream_identity_clean.csv
```

Validarlo contra el sample oficial:

```powershell
python scripts/validate_submission.py
```

El comando comprueba columnas y orden, 1.000 IDs exactos, duplicados, valores
vacíos y etiquetas legales. No lee labels de VALID.

Para reconstruir la submission desde los datos oficiales y exigir igualdad byte
a byte con el CSV aprobado:

```powershell
python scripts/prepare_data.py
python scripts/reproduce_stream_identity_submission.py
```

El segundo comando usa un directorio de trabajo nuevo, verifica los hashes del
freeze, reajusta únicamente con TRAIN, predice sin abrir labels de VALID y escribe
`outputs/predictions/submission_stream_identity_clean_regenerated.csv`. Falla si
el resultado difiere de la referencia o si intentaría sobrescribir un artefacto.
Es un entrenamiento completo y se ejecuta manualmente, nunca en CI.

El runner de investigación original
[`scripts/run_stream_identity_clean.py`](scripts/run_stream_identity_clean.py)
permanece byte a byte intacto porque forma parte de `source_sha256`. El lanzador
de reproducción aporta compatibilidad con la rama de integración sin alterar la
inferencia congelada.

## Pipeline y estructura

```text
src/ubs_recurrence/                  pipeline Stream Identity y contratos
scripts/run_stream_identity_clean.py runner de investigación congelado
scripts/reproduce_stream_identity_submission.py reproducción en integración
scripts/validate_submission.py       validator del CSV final
configs/stream_identity_clean_frozen.json receta y hashes aprobados
tests/                               tests rápidos de contrato y reproducibilidad
reports/stream_identity_clean_protocol.md protocolo y resultados
outputs/predictions/                 submission limpia versionada
```

La tarea oficial predice una etiqueta por cliente para los 90 días posteriores a
`2026-01-01`: `cloud`, `gym`, `insurance`, `mobile`, `music`, `software`,
`streaming` o `none`. La métrica es macro-F1 sobre las ocho clases. El contrato
detallado está en [`docs/OFFICIAL_CHALLENGE.md`](docs/OFFICIAL_CHALLENGE.md).

## Herramientas manuales

El benchmark histórico V1/V2 frente a Stream Identity sigue disponible como
herramienta manual:

```powershell
python -X utf8 scripts/run_ubs_stream_identity.py --run-name benchmark_train_only_20260925 --device cuda --replicas 2
```

Los scripts de auditoría e investigación bajo `scripts/` se conservan como
evidencia ejecutada. No son todos entrypoints independientes. Los runs completos,
modelos y datos permanecen ignorados por Git.

## Historia del proyecto

El repositorio empezó con pipelines V1/V2/V3/V4. Sus resultados, decisiones y
limitaciones siguen documentados en los reportes versionados, especialmente:

- [`reports/final_results.md`](reports/final_results.md)
- [`reports/stream_identity_benchmark.md`](reports/stream_identity_benchmark.md)
- [`reports/leakage_audit.md`](reports/leakage_audit.md)
- [`reports/research_decisions.md`](reports/research_decisions.md)
- [`reports/data_forensics.md`](reports/data_forensics.md)

Esos runners no son la baseline activa. El score histórico Stream Identity de
0.619493 estuvo condicionado por selección que consultó VALID y se conserva solo
como contexto. La baseline limpia congelada es la referencia actual.

## CI y contribución

GitHub Actions ejecuta `pytest`, `ruff check .` y `ruff format --check .` con
Python 3.12. El training y los benchmarks quedan como tareas manuales para evitar
coste y accesos indebidos a VALID. Antes de un cambio, consulta
[`CONTRIBUTING.md`](CONTRIBUTING.md) y no modifiques la receta congelada de forma
silenciosa.
