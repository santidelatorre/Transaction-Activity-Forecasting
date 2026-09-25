# Handoff Jaime — producto V4 / predictor V3-A

Rama de entrega: `product/v4-agent-demo-jaime`. No se cambió el modelo congelado
ni se entrenó un candidato competitivo nuevo. Se recuperaron piezas del frontend
de `origin/feature/jaime-unified-v2` (commit `16db2e9`) y se integró una sola ruta.

## Auditoría inicial y resolución de BASE_*

El encargo contenía placeholders. Se actualizó `origin` y se comprobó que la
rama partía de `origin/main`, SHA `051ce64a7cf0ab999f8aacb81fa405d5fa0257cf`,
que promueve V3-A. No se incorporaron ramas experimentales ni cambios de modelo.

| Campo | Valor resuelto desde el repositorio |
| --- | --- |
| BASE_BRANCH | `main` en la rama de partida |
| BASE_SHA | `051ce64a7cf0ab999f8aacb81fa405d5fa0257cf` |
| Commit de promoción | `fe6237eba96963f85fa643e6de8cd15cba1c67bf` |
| BASE_RUNNER | `scripts/run_ubs_v3.py`, `V3Model`, candidato `A` |
| BASE_MACRO_F1 | VALID `0.42411109773651157`, reproducido en esta integración |
| Otra métrica | Accuracy VALID `0.461`; no confundir con Macro-F1 |
| Modelo servido | V3-A, ajuste TRAIN+VALID, inferencia TEST |
| Corte / horizonte | `2026-01-01` / 90 días |

| Área auditada | Hallazgo y resultado |
| --- | --- |
| Frontend en la rama | No existía `frontend/`; estaba en la rama integrada anterior de Jaime. Se reutilizaron Vite, React, estilos, primitives, etiquetas y timeline. Se añadió una página enfocada en el caso. |
| Carpeta Laravel local | `laracopilot-fluxfuel21/` es trabajo preexistente sin seguimiento, con fixtures ficticios. No se modificó ni se añadió al commit; no interviene en la demo. |
| API anterior de Jaime | `service.py` leía `configs/ubs_v1.toml`, submission V1 y mensajes de arranque V1. No se copió esa selección obsoleta; la nueva API usa un único adaptador con lock. |
| Artefactos locales V3 | `outputs/metrics/ubs_v3/frozen_selection.json` todavía decía `full`, anterior a la promoción; su `submission_v3.csv` no era A. Se dejaron intactos y se preparó un bundle nuevo en `outputs/demo/v3a`. |
| Display de versión | Producto V4 y modelo V3-A separados; base SHA y horizonte visibles. `/api/v1` sólo versiona el contrato HTTP. No se presenta un modelo V4 inexistente. |
| README / informes | README apunta a la demo actual y marca runners antiguos como históricos. La síntesis V3 contenía «keep V2 as production baseline» y A «review only»; se añadió nota de vigencia histórica sin reescribir resultados anteriores. |
| Arranque | `.venv/Scripts/python.exe` local estaba roto. Se usó Python 3.11.16 del entorno `tx-forecasting`. Se instaló Node 22.22.0 local en `.venv`, ignorado. Vite compila y FastAPI sirve el build. |

## Qué funciona

- Adaptador único con las seis operaciones solicitadas. `predict_client` ejecuta
  `V3Model.predict_components` real, selecciona A y comprueba scores contra el
  artefacto del runner; después cachea por cliente. No usa fixtures ni valores
  sustitutos ante un fallo.
- Lock de fuentes del predictor mediante blobs Git; metadatos y huellas SHA-256
  de entradas/artefactos y versiones del entorno. No se cambió ningún archivo
  de `ubs/` ni el runner oficial. Los nuevos scripts son wrappers de integración.
- API con health, metadata, metrics, cases, client, prediction e investigation;
  404 para cliente desconocido y 503 si falta/falla el bundle. Sin endpoint de
  escritura de predicciones ni de submission.
- Evidencia histórica real: observaciones, intervalos y dispersión, variabilidad
  de importes por moneda, identidad desconocida y descripciones genéricas.
  Las descripciones se muestran normalizadas por el loader del predictor.
- Ocho herramientas de lectura y router condicional. Límite máximo de ocho
  herramientas, sin repetirlas. Parada por evidencia suficiente, límite de
  evidencia, presupuesto o selección inválida. Una política externa implementa
  `DecisionPolicy.next_tool(observations, available)` y se pasa a `investigate`;
  no recibe el adaptador ni acceso a la submission. Se descartan cambios de
  estado propuestos por ella y no se aceptan hechos narrativos libres.
- React contra API real, métricas diferenciadas, score sin calibrar, alternativas,
  candidatos/timeline, incertidumbre e investigación con conclusión.
- Comando de submission independiente, con validación de columnas, orden,
  1.000 IDs, labels y SHA; no se ha enviado nada al organizador.

## Casos reales seleccionados sin TEST labels

| Caso | Predicción | Score | Margen | Investigación |
| --- | --- | --- | --- | --- |
| `C002229` — señal clara | mobile | 0.6197056342 | 0.5476576831 | 3 herramientas: prediction, quality, recurrence |
| `C000796` — ambiguo | music | 0.3154223675 | 0.0000288054 | 6 herramientas: prediction, quality, recurrence, history, candidates, alternatives |

El claro tiene una secuencia de apoyo, pero también identidad degradada. No se
oculta esa limitación: ambos casos terminan con `evidence_limit`. La prueba
unitaria del caso con score/margen altos y evidencia sin alertas verifica que
puede parar tras una sola herramienta. Los umbrales del router son de producto,
no calibración ni ajustes a las predicciones. No se seleccionaron casos por
acierto ni usando labels TEST; sólo scores e historial previo al corte.

## Comandos de arranque

Desde la raíz y un entorno Python funcional, con datos oficiales locales:

```powershell
python -m pip install -e ".[demo,dev]" -c configs/demo_constraints.txt
python scripts/prepare_product_demo.py
cd frontend
npm ci
npm run build
cd ..
python -m uvicorn transaction_forecasting.api.main:app --host 127.0.0.1 --port 8000
```

Otra terminal: `python scripts/demo_healthcheck.py`.
Submission separada: `python scripts/build_frozen_submission.py`.
Ambos comandos de generación rechazan sobrescribir entregables previos; para
repetir se eligen rutas nuevas según `DEMO_README.md`. En esta máquina el bundle
y el CSV ya están generados: sólo hace falta arrancar el servidor.

Dependencias verificadas: Python 3.11.16; numpy 2.3.5, pandas 2.3.3,
scikit-learn 1.9.1, CatBoost 1.2.10, joblib 1.6.0; FastAPI 0.141.1,
Uvicorn 0.53.0, HTTPX 0.28.1. Node 22.22.0, Vite 8.3.1; dependencias exactas
de frontend en `package-lock.json`. La preparación no usa unlabeled ni API externa.

## Verificación realizada

- Suite Python general: **100 passed, 1 skipped** (la prueba de datos locales
  está desactivada por defecto).
- Integración con `DEMO_REAL_TESTS=1`: **4 passed**. Se invocó la función
  `predictions()` del runner oficial: paridad de los 1.000 ganadores A, más API
  y scores sobre los dos casos y seis clientes adicionales, SHA, historial,
  desconocidos y no modificación de artefactos por el agente.
- Prueba de evidencia: separación de monedas, exclusión de refunds, intervalos
  calculados, desconocidos conservados y ausencia de valores futuros inventados.
- Healthcheck HTTP real: listo, SHA correcto y recorrido completo de ambos casos.
- `npm run build`: correcto. HTML y assets servidos por FastAPI.
- `npm run test:smoke`: correcto contra API real; render de React en jsdom,
  cambio de cliente, rutas condicionales de 3/6 herramientas y error 404 sin
  presentar una predicción anterior. No es una prueba visual de píxeles.
- Ruff check y format del repositorio: correctos.
- `pre-commit run --all-files`: Ruff, formato y YAML correctos. Windows
  AppControl bloqueó los lanzadores de `trailing-whitespace` y
  `check-added-large-files` con `WinError 4551`; los mismos módulos Python
  `pre_commit_hooks.trailing_whitespace_fixer` y
  `pre_commit_hooks.check_added_large_files --enforce-all` pasaron sobre todos
  los archivos versionados. No se cambió la política de Windows. Para el commit
  se omiten sólo esos dos lanzadores, después de verificar sus controles.
- Starlette emite una advertencia de deprecación por HTTPX en TestClient; las
  pruebas pasan con las versiones registradas.

Artefactos locales ignorados, no enviados ni versionados:

| Artefacto | SHA-256 |
| --- | --- |
| `outputs/demo/v3a/model.joblib` | `a05270a70e8a9411339df8397caa04b9e11c5eb38f3292fa42d8aa9575f951ea` |
| `outputs/predictions/submission_v3a_product.csv` | `6b2420f879e1ae84987e7f3e0ee8c148d0d752982d7ba9a0e66f47b7cf9f011c` |

El CSV coincide byte a byte con el artefacto A revisado antes de esta tarea.
El hash del modelo identifica esta serialización; un nuevo fit puede contener
metadatos internos distintos aunque las predicciones coincidan.
El campo `product_commit` del bundle local registra el HEAD existente durante
la generación, anterior al commit de estos wrappers. La identidad del modelo
se verifica de forma independiente mediante su SHA base y todos sus blobs de
fuente; el código del producto reproducible es el commit de esta entrega.

## Qué no está integrado / límites

No hay LLM conectado, calibración, atribuciones individuales, identidades de
merchant verificadas, balances, predicción de fechas/importes exactos ni acciones
bancarias. No hay evaluación independiente nueva ni score de TEST etiquetado.
El servicio es una demo local sin autenticación; no está preparado para exponerlo
en Internet. Los datos disponibles son los del reto, no una integración bancaria.

No había navegador conectado disponible en la herramienta de UI (inventario
vacío); por tanto no se tomaron capturas ni se verificó visualmente el layout.
Estado listo para capturas manuales: `http://127.0.0.1:8000/?client_id=C002229`
y `http://127.0.0.1:8000/?client_id=C000796`, pulsar **Investigate this prediction**.
No se produjo vídeo. Los pitches de 60/120 s y el Q&A están en `reports/pitch/`.
