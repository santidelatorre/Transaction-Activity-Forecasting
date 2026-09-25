# Recurring Insights

Dashboard React en `src/App.jsx`, con Tailwind y gráficos SVG locales. No requiere
servicios de gráficos, claves API ni datos simulados. La evaluación corresponde al
brazo A de la baseline V3-A de `origin/main` (`051ce64`).

Desde la raíz del repositorio, en una terminal PowerShell con el entorno Python:

```powershell
conda activate tx-forecasting
# Opcional: reutilizar artefactos reales guardados en otro worktree.
$env:RECURRING_ARTIFACT_ROOT='C:\Users\jagui\Transaction-Activity-Forecasting'
python -m uvicorn transaction_forecasting.api.main:app --app-dir src --port 8001
```

En otra terminal, desde `frontend`, con Node 22.12+ disponible:

```powershell
npm ci
$env:RECURRING_API_TARGET='http://127.0.0.1:8001'
npm run dev
```

Abrir `http://127.0.0.1:5173/`. La API utiliza el puerto 8001 para poder convivir
con otra demo en 8000. El proxy usa 8000 por defecto si no se configura la variable.
Para producción: `npm run build`; FastAPI sirve ese `dist` al reiniciarse.

Con API y frontend arrancados, `npm run test:ui` comprueba en Chrome las métricas
reales, paginación, móvil, falta de datos y recuperación de errores. Genera capturas
locales en `frontend/test-results/`, ignoradas por Git. Los dos casos con datos
reales se omiten explícitamente si faltan artefactos; los de estados vacíos no.

## Datos y alcance

- `GET /api/v1/dashboard`: lee `outputs/metrics/ubs_v3/valid_results.json`,
  `importance_A.csv` y, si existe, `valid_provenance.json` bajo la raíz configurada.
- `GET /api/v1/experiments?limit=7&offset=0`: comparativas históricas del informe V3
  y, a continuación, registros existentes en SQLite; nunca crea una base de datos.
- La curva sigue el orden del informe. No se inventan fechas ni hitos de tuning.
- La cifra de clientes se calcula de la matriz de confusión. Macro-F1 y accuracy
  conservan sus nombres y significados; el resultado medido es sobre VALID, no TEST.
- Las barras corresponden a CatBoost del brazo A ajustado con TRAIN. No describen
  la contribución individual ni el ensemble completo (75 % CatBoost / 25 % heurística).
- Sin artefactos, se muestran estados vacíos; un error de API permite reintentar.
- La API de presentación no entrena, consulta TEST labels ni escribe predicciones.

Los artefactos de evaluación pueden ser históricos: su commit se muestra separado
del commit de la baseline promovida. VALID se ha reutilizado al comparar modelos.
Los endpoints heredados de clientes/submission siguen siendo V1 y no forman parte
de esta nueva ruta. La integración completa del predictor/agente V3-A permanece en
la rama de producto correspondiente.
