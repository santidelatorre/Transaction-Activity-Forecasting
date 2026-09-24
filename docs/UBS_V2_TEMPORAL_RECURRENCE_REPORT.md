# UBS V2 Temporal / Recurrence — Technical report

El informe técnico completo y autoritativo de esta fase es
[Team Handoff — Temporal / Recurrence](../reports/handoff/carles_temporal_recurrence.md).
El [resumen para la puesta en común](../reports/handoff/carles_temporal_recurrence_summary.md)
se lee en menos de dos minutos. Estos documentos analizan exclusivamente
`features/ginestar-v2-temporal`; no se creó ni se cambió a una rama report,
siguiendo la última instrucción del responsable.

## Resultado reproducido

| Protocolo | V1 Macro-F1 | Combined Macro-F1 | Delta comparable |
|---|---:|---:|---:|
| Cinco folds internos de train, media | 0,461884585 | 0,469750282 | +0,007865698 |
| Diagnóstico valid, calibración solo train | 0,140430092 | 0,153978106 | +0,013548014 |
| Runner original, selección en valid | 0,271024266 | No ejecutado con tuning V2 en valid | No comparable |

Accuracy interna: 0,5015 → 0,5060. Runner original: 0,2660 sobre 1.000
clientes. Ninguna variante gana todos los folds: no recomendar reemplazar V1.
No hay valid oficial independiente. Son hechos reproducidos, no promesas sobre
hidden test. No se cambió la submission competitiva ni el código compartido.

## Contenido del handoff

El handoff incluye alcance, commits, baseline, V2, protocolo interno/oficial,
inventario y fórmulas por feature, ausentes/soporte, estadísticas por cliente,
familia inferida y clase, análisis none y casos difíciles, todas las ablaciones,
resultados por fold, F1/precision/recall/support, matrices, errores pareados,
riesgos de cutoff/leakage, limitaciones del backtest proxy, recomendaciones
priorizadas y criterios de aceptación. Distingue observaciones, métricas,
hipótesis y experimentos propuestos TR-001/002/003.

- Evidencia ligera: [JSON completo](../reports/handoff/carles_temporal_evidence.json).
- Visualizaciones: [directorio](../reports/figures/temporal_recurrence/).
- Desarrollo previo: [informe original](UBS_V2_TEMPORAL_RECURRENCE.md), conservado.
- Base analizada: `6ba131140bc7628ff4b47360b719238f1b4add7f`.
- Main de referencia: `0199a8b7c8b2c790a9d3447b156f2088708c2864`.

## Reproducción

Desde team-repo, con el dataset original ignorado en data/raw/ubs_2026:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts/analyze_temporal_recurrence_v2.py
```

Ejecuta el runner original V1 con CSV en memoria, las seis variantes existentes,
dos cortes históricos proxy, errores pareados y figuras. Reutiliza el loader,
features y scorer actuales. Los IDs de errores se conservan solo en
`outputs/metrics/carles_temporal_handoff/private_oof_error_pairs.csv`, ignorado.
No genera submission en disco. Sin entrenar ni consultar valid de nuevo:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts/analyze_temporal_recurrence_v2.py --render-only
```

## Verificación

51 tests pasan; Ruff global y formato correctos; cuatro figuras revisadas.
El runner verifica las 1.000 filas de su CSV en memoria mediante el validador
existente. Métricas por fold idénticas al desarrollo anterior. No hay cambios
en src, configs, tests ni runner V1. El aviso NumPy de overflow volvió a aparecer
en la reproducción y se conserva en la evidencia; no se atribuye causa sin prueba.
Pre-commit y revisión final de Git completan el cierre documentado en el commit.
