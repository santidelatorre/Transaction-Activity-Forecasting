# Jaime: Dashboard + tracking V2

## Resultado y alcance

Preparado el 24 de septiembre de 2026. El parche `jaime_unified_v2.patch`, situado
junto a esta guía, se aplica **después de fusionar las dos ramas**. Incluye el
adaptador del runner, lectura SQLite, endpoint, componente React y tests.
No crea ramas, commits ni pushes automáticamente.

Referencias verificadas tras `git fetch origin`:

| Referencia local y remota | Commit |
| --- | --- |
| dev/jaime | 4bb063c649ea91b46ba758b5607009875ae740d8 |
| features/jaime-v2-experiment-tracking | f197575421cabaefa73422f493edea51c5a5afa3 |
| Base común y origin/main | 0199a8b7c8b2c790a9d3447b156f2088708c2864 |

Ambas ramas contienen el main remoto inspeccionado. No hace falta integrar main
otra vez en esta revisión. La petición explícita de Jaime determina la rama de
destino; la política personal de Santiago no cambia este destino.

## Conflictos y decisiones

| Archivos | Hallazgo y resolución |
| --- | --- |
| `.gitignore` | Único archivo modificado por ambos lados. Cambios en zonas distintas; la simulación `merge-tree` los combina sin conflicto. Conservar `frontend/node_modules/` y `outputs/experiments/*.sqlite3*`. |
| `pyproject.toml` | Solo Dashboard añade el extra `web` con FastAPI/Uvicorn. Conservarlo. SQLite no requiere dependencia adicional. |
| `README.md`, `frontend/`, `src/transaction_forecasting/api/` | Aportaciones del Dashboard; conservar. |
| `experiment_tracking.py`, sus tests y handoffs | Aportaciones del tracking; conservar. |
| `scripts/run_ubs_baseline.py` | Sin divergencia entre ramas, pero falta conectar el logger. El parche añade la llamada tras seleccionar modelo y antes del refit. |
| `scripts/format_submission.py` | Corrección pequeña de formato para que el Ruff local acepte el código heredado del Dashboard. |

La rama denominada V2 aporta tracking, no nuevos modelos V2. El runner sigue
identificando sus modelos como `ubs-v1`; no se inventa una mejora de calidad.

## Integración lógica

```text
Runner UBS -> métricas oficiales de validación -> adaptador -> ExperimentLogger -> SQLite
                                                                             |
React TechnicalResults <- GET /api/v1/experiments <- lector SQLite mode=ro <----+

Runner UBS -> summary.json / experiments.csv / submission.csv -> API V1 existente
```

- `tracking_integration.record_validation_run` convierte `per_class[label]['f1-score']`
  en `f1_per_class[label]`, requerido por el logger. Registra configuración,
  semilla incluida en ella, notas del candidato y descripciones de los conjuntos
  de features. Estas descripciones no son un inventario de columnas transformadas.
- Todos los candidatos de una ejecución comparten `metrics.run_id`; cada registro
  conserva su UUID. `selected` significa seleccionado dentro de esa ejecución.
- El tiempo almacenado suma entrenamiento e inferencia; excluye construcción
  compartida de features y scoring. No se pasa una baseline histórica sin verificar.
- La API abre la base en modo lectura, limita las páginas a 100 registros y no
  crea una base vacía al abrir el Dashboard. Una base ausente devuelve lista vacía;
  una base corrupta produce 503 mediante el manejador existente.
- React permite refrescar y paginar el historial incluso sin `summary.json`.
  Un fallo del historial no bloquea el resto de la página.
- El resumen actual y el CSV mantienen sus contratos. El historial SQLite puede
  contener distintas ejecuciones/protocolos: no se usa su mejor F1 para sustituir
  automáticamente el modelo de la submission.
- No recalcular métricas en React, no escribir SQLite desde peticiones GET y no
  duplicar el logger. CSV es un resumen de ejecución; SQLite es el historial.
  No eliminar CSV en esta integración porque tiene consumidores existentes.

El logger guarda cada candidato en una transacción independiente. Si falla una
escritura, el runner falla explícitamente antes del refit; puede quedar un historial
parcial de candidatos ya evaluados. Reejecutar crea otro run_id. No se registra
el refit train+valid como si fuera una nueva evaluación independiente.

## Git: pasos exactos en PowerShell

Ejecutar los bloques por separado y revisar cada resultado. El worktree evita
mover o guardar en stash `.vscode/settings.json` y `laracopilot-fluxfuel21/`, que
ya estaban modificados/sin seguimiento en el workspace original.

### 1. Fijar entradas y crear un árbol aislado

```powershell
Set-Location C:\Users\jagui\Transaction-Activity-Forecasting
$patchPath = (Resolve-Path reports/handoff/jaime_unified_v2.patch).Path
git status --short --branch
git fetch origin
if ($LASTEXITCODE -ne 0) { throw 'Falló fetch' }
if ((git rev-parse origin/dev/jaime) -ne '4bb063c649ea91b46ba758b5607009875ae740d8') {
    throw 'Dashboard cambió: revisar diff y regenerar/verificar el parche'
}
if ((git rev-parse origin/features/jaime-v2-experiment-tracking) -ne 'f197575421cabaefa73422f493edea51c5a5afa3') {
    throw 'Tracking cambió: revisar diff y regenerar/verificar el parche'
}
git worktree add -b feature/jaime-unified-v2 ../Transaction-Activity-Forecasting-jaime-unified origin/dev/jaime
if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el worktree; no sobrescribir una rama existente' }
Set-Location ../Transaction-Activity-Forecasting-jaime-unified
git merge --no-ff --no-commit origin/features/jaime-v2-experiment-tracking
```

### 2. Revisar y completar la integración

```powershell
git status --short
git diff --name-only --diff-filter=U
```

En las revisiones inspeccionadas no se esperan conflictos. Si aparecen, detenerse
y resolver cada archivo; en `.gitignore` conservar ambas reglas citadas y quitar
marcadores, luego `git add .gitignore`. No usar `--ours`/`--theirs` sobre todo el
árbol. Para cancelar antes de aplicar el parche: `git merge --abort` dentro del
worktree nuevo. La copia original conserva sus cambios.

```powershell
git apply --check $patchPath
if ($LASTEXITCODE -ne 0) { throw 'Parche incompatible: no aplicar a ciegas' }
git apply $patchPath
if ($LASTEXITCODE -ne 0) { throw 'Falló la aplicación del parche' }
git diff --check
git diff --stat
```

### 3. Instalar y verificar en el worktree

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev,web]'
python -m pytest
python -m ruff check .
python -m ruff format --check .
Push-Location frontend
npm install
npm run build
Pop-Location
```

Se requiere Node/npm compatible con Vite del `package.json`. El `npm install`
generará `frontend/package-lock.json`; conservarlo para instalaciones posteriores
con `npm ci`. Revisar los resultados antes de continuar.

```powershell
git add src/transaction_forecasting/tracking_integration.py src/transaction_forecasting/api/main.py src/transaction_forecasting/api/router.py src/transaction_forecasting/api/service.py
git add scripts/run_ubs_baseline.py scripts/format_submission.py tests/test_tracking_integration.py
git add frontend/src/components/insights/ExperimentHistory.jsx frontend/src/pages/TechnicalResults.jsx frontend/package-lock.json
python -m pre_commit run --all-files
```

Si los hooks reformatean archivos, revisar `git diff`, añadir explícitamente los
archivos corregidos y repetir los checks. El hook fija Ruff 0.6.9; el Ruff del
entorno puede diferir. No crear el commit si hay errores pendientes.

### 4. Revisar, guardar y publicar

```powershell
git diff --cached --check
git diff --cached --stat
git status --short
git commit -m 'Integrate Jaime dashboard with SQLite experiment history'
git push -u origin feature/jaime-unified-v2
```

El commit completa el merge y conserva ambas historias. El push es a la nueva
rama, nunca a main. Abrir PR hacia `dev/jaime` para reunir el trabajo personal;
la posterior integración en main sigue el PR y revisión acordados por el equipo.
No se ha ejecutado ninguno de estos pasos de publicación durante la preparación.

## Entrada principal unificada

No hace falta un `app.py` monolítico: el router conecta los servicios. El parche
deja `src/transaction_forecasting/api/main.py` así:

```python
"""FastAPI app for the React forecasting dashboard."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from transaction_forecasting.api.router import router
from transaction_forecasting.api.service import PROJECT_ROOT, ArtifactUnavailable

app = FastAPI(
    title="Transaction Activity Forecasting",
    description="Dashboard API for UBS predictions, validation artifacts and experiment history.",
    version="0.2.0",
)
app.include_router(router)


@app.exception_handler(ArtifactUnavailable)
async def artifact_error_handler(_request, exception: ArtifactUnavailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exception)})


frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
```

### Demo

Los datos y outputs ignorados no se copian al crear un worktree. Preparar los datos
oficiales localmente según `configs/ubs_v1.toml`, sin añadirlos a Git. Después:

```powershell
python scripts/run_ubs_baseline.py --config configs/ubs_v1.toml
python -m uvicorn transaction_forecasting.api.main:app --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000` después del build y navegar a Technical results.
El frontend usa hash routing; mantener su navegación existente. Mostrar resumen
de validación, historial, commit/configuración y estado de selección. Sin datos,
mostrar el estado vacío; no presentar fixtures de tests como resultados reales.

## Verificación realizada y límites

- Snapshot aislado con ambos árboles y el parche: **45 tests pasan**, incluyendo
  round-trip runner/adaptador/logger/API-service, paginación, DB ausente/corrupta
  y la batería existente del logger.
- Ruff check, Ruff format y compilación Python pasan en ese snapshot.
- Parche validado por aplicación inversa en modo check contra el snapshot final.
- No se verificó HTTP real ni build/navegación React: FastAPI y Node/npm no están
  disponibles en el entorno usado para esta revisión.
- Pre-commit no se completó: su caché global es de solo lectura en este entorno.
  Ejecutarlo en el worktree real antes del commit.
- El `.venv` original apunta a un Python inexistente. Se usó el Python 3.11.16
  disponible en el entorno `tx-forecasting`; crear el nuevo `.venv` como arriba.
- No se entrenó con datos oficiales ni se verificó una nueva submission.
  Los tests demuestran integración del software, no calidad predictiva.
