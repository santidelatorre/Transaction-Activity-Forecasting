# Swiss AI Weeks / Transaction Activity Forecasting

## Contrato oficial y trabajo actual — 24 septiembre 2026

El objetivo oficial ya está confirmado: predecir la próxima familia recurrente
por cliente (`cloud`, `gym`, `insurance`, `mobile`, `music`, `software`,
`streaming`, `none`), corte `2026-01-01`, horizonte 90 días, métrica macro-F1.
El [contrato oficial](docs/OFFICIAL_CHALLENGE.md) sustituye las hipótesis
provisionales de las secciones históricas siguientes. La [propuesta de producto
e integración](docs/IDEAS_AND_INTEGRATION.md) compara las ideas del equipo y las
dos bases de código.

El smoke pipeline actual sigue siendo MOCK y no mide calidad predictiva.
Ya se pueden puntuar predicciones oficiales y comprobar el CSV de entrega:

```powershell
python -m transaction_forecasting.evaluation.official score --labels data/raw/valid_labels.csv --predictions outputs/predictions/valid.csv
python -m transaction_forecasting.evaluation.official validate --sample data/raw/sample_submission.csv --predictions outputs/predictions/submission.csv
```

Requiere el paquete instalado según las instrucciones de entorno de abajo.
El adaptador y entrenamiento UBS V1 ya están implementados en `ubs/` y
`scripts/run_ubs_baseline.py`, con configuración en `configs/ubs_v1.toml`.
Los datos se descargan localmente y no se versionan. Conservar el split
train/valid/test suministrado; no sustituirlo por un split de filas.
El runner UBS y el CLI oficial comparten cálculo de métricas y validación de
entrega; UBS conserva además su requisito de orden idéntico al sample.

Acuerdo actual comunicado por Carles: cada persona trabaja en su rama
`dev/<nombre>` y propone cambios mediante PR a `main`, sin hacer merge automáticamente.

## Contexto inicial y entorno

El proyecto incluye tres recorridos: smoke con datos ficticios, pipeline genérico
configurable y clasificación UBS V1 sobre el esquema oficial. Los dos primeros
siguen siendo herramientas de desarrollo; el runner UBS es la entrada para
entrenar y generar predicciones del challenge.

## Hackathon quick start

The repository now includes a small deterministic end-to-end smoke path that is
for quick development checks, separate from the official UBS training workflow:

```bash
python -m pytest
python -c "from transaction_forecasting.pipeline import run_smoke_pipeline; print(run_smoke_pipeline()[1])"
python -m transaction_forecasting.pipeline --synthetic
```

## Pipeline configurable

El flujo genérico se ejecuta con `python -m transaction_forecasting.pipeline
--config configs/default.toml`. Mientras no se conozca el dataset, usa
`--synthetic`, que no representa ninguna columna ni regla de negocio real.

`configs/default.toml` centraliza los campos marcados `TODO(dataset)`: ruta y
formato, contrato de columnas/tipos, fecha de corte, horizonte/embargo, target
y parámetros. El pipeline carga CSV o Parquet, limpia duplicados exactos, hace
el split temporal train/validation/test, ajusta transformadores solo con train,
evalúa candidatos y guarda registros JSONL. El baseline genérico es una media
del target y existe solo para verificar el flujo hasta que se defina la tarea.

## UBS baseline V1

El baseline real del challenge crea una fila por cliente, aprende asociaciones de
descripciones solo con train y compara dummy, recurrencia, regresión logística,
CatBoost y un ensemble simple. Se ejecuta desde la raíz del repositorio:

```powershell
python scripts/run_ubs_baseline.py --config configs/ubs_v1.toml
```

El comando usa train para todos los ajustes supervisados, valid para evaluación y
selección, y test únicamente para generar
`outputs/predictions/submission_v1.csv`. Después de seleccionar el enfoque, lo
reajusta con train+valid para producir la submission. Los resultados detallados
quedan en `outputs/metrics/ubs_v1/`; datos y outputs permanecen ignorados por Git.

The shared contracts, workstream split, temporal evaluation rules, and milestone
plan are in [`docs/HACKATHON_PLAN.md`](docs/HACKATHON_PLAN.md),
[`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md), and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Local development setup

The project targets Python 3.11. Python 3.12 is a compatible fallback when 3.11 is not available locally.

Windows PowerShell (recommended):

```powershell
git clone https://github.com/santidelatorre/Transaction-Activity-Forecasting.git
cd Transaction-Activity-Forecasting
.\scripts\setup.ps1
```

Activate the environment manually when needed with `.\.venv\Scripts\Activate.ps1`.

Linux/macOS:

```bash
git clone https://github.com/santidelatorre/Transaction-Activity-Forecasting.git
cd Transaction-Activity-Forecasting
./scripts/setup.sh
source .venv/bin/activate
```

The setup script creates the editable package, installs development dependencies, registers the `transaction-forecasting` Jupyter kernel, and installs pre-commit hooks. It is safe to run again. If PowerShell blocks local scripts, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

In VS Code, open the repository folder and select `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Linux/macOS) if it is not selected automatically. Run tests and quality checks with:

```powershell
pytest
ruff check .
ruff format --check .
pre-commit run --all-files
```

Start Jupyter with `jupyter lab`. Install a new dependency by adding it to `pyproject.toml`, then rerun `python -m pip install -e ".[dev]"`. The local `.env` file is optional; copy `.env.example` to `.env` and never commit secrets or challenge datasets.

## Estructura

- `data/`: datos locales (`raw`, `interim`, `processed`, `external`), nunca versionados.
- `notebooks/`: exploración y experimentación por etapas (`00_sandbox` a `05_evaluation`).
- `src/transaction_forecasting/`: paquetes independientes para datos, preprocessing, recurrencia, features, modelos, evaluación, visualización y utilidades.
- `configs/`: configuración ligera y versionable.
- `scripts/`: puntos de entrada pequeños.
- `tests/`: tests automatizados.
- `outputs/`: figuras y resultados generados localmente.
- `docs/`: decisiones y documentación complementaria.

## Instalación

Se requiere Python 3.11 o compatible:

```bash
git clone <URL_DEL_REPOSITORIO>
cd transaction-activity-forecasting
python -m venv .venv
```

Windows PowerShell: `\.venv\Scripts\Activate.ps1`

Linux/macOS: `source .venv/bin/activate`

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

Comprobaciones locales:

```bash
ruff check .
ruff format --check .
pytest
```

## Workflow Git

No trabajes directamente sobre `main`:

```bash
git switch dev/carles
git fetch origin
git merge origin/main
```

El ejemplo supone que tu rama personal ya existe. Usa tu nombre en lugar de `carles`.
Añade archivos concretos con `git add`, crea un commit y usa
`git push -u origin dev/carles`. Un PR propone cambios a `main`; no los integra
automáticamente. Consulta [CONTRIBUTING.md](CONTRIBUTING.md) para crear o recuperar
tu rama por primera vez.

## Notebooks y datos

Los notebooks se usan para EDA y experimentos; la lógica reutilizable vive en
`src/`. El dataset oficial se obtiene de UBS y se conserva localmente en
`data/raw/ubs_2026`, según `configs/ubs_v1.toml`. No se suben datasets a Git.

## Filosofía de modelado

La comparación prevista es progresiva: baseline estadístico; feature engineering y detección de recurrencia; CatBoost/LightGBM/XGBoost; y modelos secuenciales o deep learning solo si aportan valor medible. PyTorch y `sentence-transformers` podrían añadirse más adelante para secuencias y embeddings, pero no son dependencias iniciales.

La evaluación debe respetar la causalidad: historial pasado → evento futuro. Se evitará el leakage temporal y no se usará un split aleatorio salvo justificación explícita.

## Reproducibilidad y colaboración

`configs/default.toml` contiene rutas y semilla inicial. Las utilidades estándar controlan logging, `random` y NumPy. Con siete desarrolladores, mantén módulos pequeños, responsabilidades separadas y Pull Requests acotados; no mezcles EDA, features y modelos en un mismo cambio.

La integración en `main` se propone mediante Pull Requests y requiere revisión
y CI. La configuración de protección de rama y los permisos los gestiona el
propietario del repositorio. No se presupone una licencia ni permiso adicional
de publicación de datos por disponer de este código.

## UBS V2 integrada

La receta V2 congelada combina 75% CatBoost sobre 146 agregados históricos sin
features derivadas del target y 25% heurística de periodicidad. En el mismo split
oficial obtiene Macro-F1 **0.391549456** y accuracy **0.4240**, frente a
**0.271024266 / 0.2660** de V1. La validación ha sido reutilizada para selección;
`music` sigue por debajo de V1. Consulta el
[informe completo](reports/v2_final_report.md) y el
[resumen del equipo](reports/v2_final_summary.md).

Desde la raíz del repositorio, con el entorno instalado:

```powershell
python scripts/run_ubs_v2.py
```

Genera `outputs/metrics/ubs_v2/` y
`outputs/predictions/submission_v2.csv`, ajustando primero solo con train para
validación y después con train+valid para test. V1 conserva su runner,
configuración y submission independientes. En la máquina de integración se
recuperó Python en `.venv/runtime_v2/python.exe`, que puede sustituir a `python`
si el ejecutable original de `.venv` apunta a una instalación eliminada.

Los experimentos rechazados solo se ejecutan mediante selección explícita en
`scripts/evaluate_v2_candidate.py`; el runner final usa la receta congelada.
`scripts/audit_ubs_v2.py` añade cinco folds internos y análisis pareado del
ledger local. Datos, predicciones, modelos, cachés y SQLite permanecen ignorados
por Git.
