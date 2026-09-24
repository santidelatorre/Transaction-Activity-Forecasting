# V1 — Team Summary

V1 COMMIT: 0199a8b7c8b2c790a9d3447b156f2088708c2864

V2 COMMIT: 96409b7940a991fbda5235b85ffeb40652a4087b

V1 HISTORICAL MACRO-F1: 0.2710243

V1 REPRODUCED MACRO-F1: 0.271024266

V1 REPRODUCED ACCURACY: 0.266000

V2 COMPARABLE MACRO-F1: 0.391549456

DELTA: +0.120525190

## V1 main components

- Mapa supervisado descripción/familia ajustado con train.
- Heurístico de recurrencia con 32 entradas familiares; bias none=-1, T=1.
- Selección sobre valid y refit del mapa sobre train+valid para test.

## Three strengths

1. Baseline reproducido desde un commit fijo.
2. Métrica oficial de ocho clases y alineación por ID.
3. Submission con contrato verificado y receta interpretable.

## Three weaknesses

1. Sesgo de selección por reutilizar valid.
2. none mal detectada; exceso de predicciones de algunas familias.
3. Candidatos ML originales con codificación influida por la propia etiqueta.

## What V2 improves

- Macro-F1 y accuracy suben bajo el mismo protocolo.
- 246 errores V1 corregidos frente a 88 aciertos V1 perdidos.
- none presenta la mayor mejora de F1.

## What V2 worsens

- Music pierde F1; la mejora global no beneficia a todas las clases.

## Main validation limitation

Valid se reutilizó en selección. No hay resultado oculto deducible del CSV test.

## Submission validation

Ambas reproducciones: 1.000 IDs únicos, columnas/clases/orden válidos.
El archivo V2 realmente enviado al dashboard no fue adjuntado en esta tarea.

## Recommendation

Conservar V1 como control y usar V2 para revisión del equipo por su mejora local.
Ver `v1_final_report.md`, `v1_vs_v2_comparison.md` y `v1_benchmark_checks.md`.
