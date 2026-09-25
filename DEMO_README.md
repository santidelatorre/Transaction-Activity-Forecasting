# Recurring Insights — integración V4, predictor V3-A TRAIN-only

La síntesis conserva V3-A. La versión de receta es
`v4-synthesis-v3a-train-only-1`; el producto y el agente usan exactamente ese
predictor. No hay refit TRAIN+VALID, cambios de threshold ni selección por la demo.

## Preparación

Python 3.11–3.13, dependencias del proyecto y datos oficiales locales en
`data/raw/ubs_2026/`. Los datos, modelos, scores y submissions están ignorados.

```powershell
python -m pip install -e ".[demo,dev]" -c configs/demo_constraints.txt
python scripts/run_ubs_v4.py --phase valid
python scripts/run_ubs_v4.py --phase submission
python scripts/prepare_product_demo.py
```

El modelo se ajusta una vez con TRAIN; VALID se lee después de persistir las
predicciones. La fase submission no lee ni hashea VALID labels. El bundle copia
el mismo modelo serializado y comprueba paridad de los 1.000 scores TEST.
Sus métricas son de VALID reutilizado, nunca de TEST.

Los resultados están en `outputs/metrics/ubs_v4_final/` y el bundle en
`outputs/demo/v4_train_only/`. Los comandos rechazan sobrescribir entregables o
reutilizar una procedencia incompatible. Para repetir, usar directorios nuevos
con `--output-dir` y `--runner-dir`; establecer `DEMO_BUNDLE` al bundle elegido.

## Interfaz

```powershell
cd frontend
npm ci
npm run build
cd ..
python -m uvicorn transaction_forecasting.api.main:app --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000`. El frontend requiere Node >=22.12. Verificar:

```powershell
python scripts/demo_healthcheck.py
$env:DEMO_REAL_TESTS='1'
python -m pytest tests/test_product_integration.py
cd frontend
npm run test:smoke
```

Esta máquina no dispone de FastAPI y la instalación está bloqueada; npm tampoco
ha podido recuperar las dependencias. Por tanto, no se declara verificación HTTP,
build React, smoke de navegador ni inspección visual en la integración actual.
La evidencia de esta ronda se detalla en `reports/v4_synthesis.md`.

## Contrato y límites

API: `/api/v1/health`, `/model`, `/metrics`, `/cases`, `/clients/{id}`,
`/clients/{id}/prediction` e investigación POST `/clients/{id}/investigate`.
Cliente desconocido: 404; artefacto ausente o incompatible: 503.

Jaime aporta ocho herramientas y una política condicional determinista, sin LLM:
`predict_client`, `inspect_history`, `inspect_data_quality`,
`inspect_candidate_streams`, `inspect_recurrence`, `compare_alternatives`,
`show_model_metadata`, `show_global_metrics`. El presupuesto máximo es ocho
herramientas distintas. No existe herramienta de escritura de submission.

Los casos se eligen por margen y evidencia histórica, sin TEST labels; los IDs
cambian con la receta TRAIN-only. Mostrar versión, horizonte de 90 días,
ambigüedad y scores sin calibrar. No afirmar identidad de comercio verificada,
saldo, causalidad ni fecha/importe futuro exactos. La demo es local, sin
autenticación y no preparada para exposición pública.

La submission oficial es `outputs/metrics/ubs_v4_final/submission_v4.csv`.
`scripts/build_frozen_submission.py` puede exportar el mismo predictor desde el
bundle a otra ruta, sin agente. Ningún comando envía el CSV al organizador.
