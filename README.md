# Swiss AI Weeks / Transaction Activity Forecasting

Base de proyecto para un hackathon de AI/ML. El objetivo provisional es predecir la siguiente transacción recurrente de un cliente a partir de su historial. El esquema, el objetivo exacto y las reglas de evaluación pueden cambiar cuando recibamos el dataset y la documentación oficial.

Esta fase prepara infraestructura; no incluye datos reales, no asume columnas concretas y no entrena modelos.

## Hackathon quick start

The repository now includes a small deterministic end-to-end smoke path that is
safe to use while the challenge dataset and official metric are being confirmed:

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
`outputs/predictions/submission_v1.csv`. Los resultados detallados quedan en
`outputs/metrics/ubs_v1/`; datos y outputs permanecen ignorados por Git.

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
git checkout main
git pull
git checkout -b feat/nombre-descriptivo
```

Después: `git add .`, `git commit`, `git push -u origin feat/nombre-descriptivo` y abre un Pull Request hacia `main`. Para este hackathon, usa las ramas concretas de `docs/HACKATHON_PLAN.md`.

## Notebooks y datos

Los notebooks se usan para EDA y experimentos; la lógica reutilizable debe moverse a `src/` y no debe construirse el pipeline completo dentro de notebooks. No se suben datasets al repositorio. Las carpetas de datos se conservan con `.gitkeep`; cuando conozcamos el dataset definiremos un mecanismo común y seguro para compartirlo.

## Filosofía de modelado

La comparación prevista es progresiva: baseline estadístico; feature engineering y detección de recurrencia; CatBoost/LightGBM/XGBoost; y modelos secuenciales o deep learning solo si aportan valor medible. PyTorch y `sentence-transformers` podrían añadirse más adelante para secuencias y embeddings, pero no son dependencias iniciales.

La evaluación debe respetar la causalidad: historial pasado → evento futuro. Se evitará el leakage temporal y no se usará un split aleatorio salvo justificación explícita.

## Reproducibilidad y colaboración

`configs/default.toml` contiene rutas y semilla inicial. Las utilidades estándar controlan logging, `random` y NumPy. Con siete desarrolladores, mantén módulos pequeños, responsabilidades separadas y Pull Requests acotados; no mezcles EDA, features y modelos en un mismo cambio.

En GitHub se deberá proteger `main`, exigir Pull Requests y requerir que CI pase antes del merge. La URL real del repositorio, licencia y política definitiva de acceso a datos quedan pendientes de configuración del equipo.
