# Recurring Insights: producto V4, modelo congelado V3-A

Demo local de una ruta: cliente → predicción → evidencia → investigación.
No necesita claves API. El agente es una política determinista condicional;
no se presenta como LLM. No cambia predicciones ni genera entregas.

## Arranque reproducible

Requisitos: Python 3.11 o 3.12 y Node.js >=22.12 (probado con 22.22.0).
Desde la raíz, en un entorno Python funcional:

```powershell
python -m pip install -e ".[demo,dev]" -c configs/demo_constraints.txt
```

Colocar los archivos oficiales, sin modificarlos, en `data/raw/ubs_2026/`:
`train_transactions.jsonl`, `train_labels.csv`, `valid_transactions.jsonl`,
`valid_labels.csv`, `test_transactions.jsonl`, `sample_submission.csv`.
No se necesita el archivo unlabeled. No se leen etiquetas TEST.
Los datos y modelos no están incluidos en Git.

1. Ejecutar el predictor congelado (varios minutos, CPU):

```powershell
python scripts/prepare_product_demo.py
```

Usa el `V3Model` sin modificaciones, receta A, del runner
`scripts/run_ubs_v3.py`. Mide VALID con TRAIN y ajusta TRAIN+VALID para TEST.
Comprueba las 1.000 predicciones contra `V3Model.predict`, guarda el modelo,
scores, métricas medidas, casos y fingerprints en `outputs/demo/v3a/`.
No ejecuta OOF ni selecciona otra receta. El runner histórico completo sigue
disponible; no reutilizar `outputs/metrics/ubs_v3`, que localmente era anterior
a la promoción de A. La preparación exige un directorio nuevo: para repetir,
usar `--output-dir outputs/demo/v3a-repeat` y establecer `DEMO_BUNDLE` a esa ruta.
Un directorio sin `manifest.json` indica una preparación incompleta.

2. Compilar el frontend recuperado de Jaime:

```powershell
cd frontend
npm ci
npm run build
cd ..
```

3. Arrancar API y frontend juntos, **después** de compilar:

```powershell
python -m uvicorn transaction_forecasting.api.main:app --host 127.0.0.1 --port 8000
```

Abrir [la demo local](http://127.0.0.1:8000). En otra terminal:

```powershell
python scripts/demo_healthcheck.py
```

El healthcheck exige el SHA correcto y recorre ambos casos, historial e
investigación. `/api/v1/health` devuelve 503 si falta o no coincide el bundle;
un cliente desconocido devuelve 404. `v1` aquí versiona el contrato HTTP,
no el modelo. Reiniciar la API si se recompila el frontend o cambia el bundle.
El modelo y sus resultados se mantienen en memoria durante el proceso.

Desarrollo opcional: mantener la API en 8000 y ejecutar `npm run dev` dentro
de `frontend`; Vite sirve en 5173 y reenvía `/api` al backend. No abrir el
`index.html` directamente. No hace falta Laravel ni PHP.

En la máquina de Jaime, `.venv/Scripts/python.exe` apunta a una instalación
eliminada. Se verificó con:
`C:/Users/jagui/miniforge3/envs/tx-forecasting/python.exe`.
Node local está en `.venv/node-runtime/node-v22.22.0-win-x64/`; añadir ese
directorio al PATH para usar `npm`. No se distribuyen estos entornos.

## Recorrido de menos de 60 segundos

- Caso claro: `/?client_id=C002229`, familia `mobile`, score `0.619706`,
  margen `0.547658`, una secuencia recurrente de apoyo. La calidad de identidad
  sigue degradada: explicar esta limitación. Investigación: 3 herramientas.
- Caso ambiguo: `/?client_id=C000796`, familia `music`, score `0.315422`,
  margen `0.000029`. Investigación: 6 herramientas; evidencia insuficiente
  para eliminar la ambigüedad. La familia oficial permanece intacta.
- Pulsar **Investigate this prediction**. Mostrar herramientas elegidas,
  conclusión y condición de parada. Los paneles de detalle son opcionales.

Los casos salen de TEST histories y scores, elegidos por margen, con recurrencia
de apoyo requerida para el claro. La función de selección no acepta etiquetas.
Son ejemplos ilustrativos de comportamiento, no una estimación de calidad.
Para capturas: ventana 1440×1100 o mayor, caso seleccionado e investigación
abierta. No hay fixtures en este recorrido. Vídeo de respaldo: grabación manual
del mismo recorrido; no se ha generado un vídeo.

## Submission: comando separado

```powershell
python scripts/build_frozen_submission.py
```

Genera `outputs/predictions/submission_v3a_product.csv` directamente mediante
el predictor real; valida columnas y orden oficiales, 1.000 IDs únicos e
idénticos al sample, labels permitidos y paridad con el bundle. Escribe
`submission_v3a_product.metadata.json` con SHA del modelo, SHA-256 del artefacto
y del CSV. No usa el agente y no envía nada al organizador. Rechaza sobrescribir
un CSV existente; para repetir: `--output outputs/predictions/submission_v3a_repeat.csv`.

## Verificación

```powershell
python -m pytest tests/test_product_agent.py tests/test_product_integration.py
$env:DEMO_REAL_TESTS = "1"
python -m pytest tests/test_product_integration.py
Remove-Item Env:DEMO_REAL_TESTS
python -m ruff check .
python -m ruff format --check .
python -m pre_commit run --all-files
```

Con la API arrancada, dentro de `frontend`: `npm run test:smoke`. Esta prueba
monta la pantalla React en jsdom y recorre ambos casos contra la API real,
incluyendo el error de cliente desconocido; no verifica píxeles ni sustituye
una revisión visual en navegador.

En bash: `DEMO_REAL_TESTS=1 python -m pytest tests/test_product_integration.py`.
La prueba real invoca `predictions()` del runner oficial y compara sus 1.000
ganadores A; comprueba API, scores, SHA, evidencia y agente en ocho clientes.
Comprueba que la investigación no cambia ningún archivo del bundle.

## Modelo y límites

Base: `main`, SHA `051ce64a7cf0ab999f8aacb81fa405d5fa0257cf`;
promoción V3-A: `fe6237eba96963f85fa643e6de8cd15cba1c67bf`.
Los `BASE_*` del encargo estaban vacíos: se resolvieron desde la baseline
promovida en la rama de partida, no desde experimentos posteriores.
Lock completo: `configs/product_model_lock.json`. Cambiar a otra baseline
requiere una nueva integración y verificación, no cambiar una etiqueta visual.

Macro-F1 VALID reproducido: **0.42411109773651157**; accuracy: **0.461**.
VALID reutilizado: no es una evaluación independiente. Scores sin calibrar.
Historial antes de `2026-01-01`; horizonte **90 días**. `none` es una clase.
Las asociaciones de familia se aprenden de labels por cliente, no de identidades
verificadas por transacción. No se calculan atribuciones causales, saldos,
importes futuros exactos o fechas garantizadas. Servicio local, sin autenticación;
mantener el bind a `127.0.0.1` para esta demo.

Referencias de herramientas: [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
y [Vite](https://vite.dev/guide/). Pitch y defensa: `reports/pitch/`.
