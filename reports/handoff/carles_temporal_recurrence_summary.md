# Temporal / Recurrence — Meeting Summary

## Baseline

Macro-F1: **0,2710243** (valid oficial, usada para selección).
Accuracy: **0,2660**, 1.000 clientes; reproducido sin escribir submission.

## Temporal V2

Macro-F1: **0,4697503**, media de cinco folds internos.
Accuracy: **0,5060**.
Difference vs V1: **+0,0078657 Macro-F1; +0,0045 accuracy**, respecto al
control interno V1 **0,4618846 / 0,5015**, no respecto al 0,271 oficial.

## 3 strongest temporal findings

1. Combined gana media, pero solo 3/5 folds: no promoción fiable.
2. Corrige 57 clientes y estropea 48; 97 cambios son entre familias.
3. Dos eventos no demuestran regularidad: 9.325 streams tienen un solo intervalo.

## 3 main weaknesses

1. Valid contaminada; no hay evaluación oficial independiente.
2. Calibración interna transfiere mal a valid: 837/1.000 none en combined.
3. Fechas proxy: MAE ~64–73 días; no etiquetas históricas oficiales por familia.

## 5 key findings

1. V1 esperada reproducida; 51 tests pasan.
2. Resultados por fold idénticos a la ejecución anterior.
3. Mejoran cinco familias; empeoran streaming, gym y none.
4. None pasa de 487 a 484 predicciones OOF; su F1 baja ligeramente.
5. Periodicity tiene mayor media, pero tampoco gana todos los folds.

## Features worth integrating

1. Guardias de cutoff y deduplicación temporal.
2. Soporte y desconocimiento explícitos, como diagnóstico.
3. Interfaz de bloques desactivables; ningún peso predictivo aprobado aún.

## Features rejected or uncertain

1. Pesos combined actuales: inestables.
2. Calendario mensual: impacto de fecha mínimo.
3. Periodicidad anual: sin soporte suficiente.

## Classes improved

1. Mobile: ΔF1 OOF **+0,032120**.
2. Music: **+0,018592**.
3. Software: **+0,011468**; también mejoran insurance y cloud.

## Classes worsened

1. Streaming: **−0,010610**.
2. Gym: **−0,002845**.
3. None: **−0,001852**.

## Top 3 recommended experiments

1. TR-001: mantener bias de V1 fijo para aislar la señal temporal.
2. TR-002: regularidad con soporte, sin cambiar otros términos.
3. TR-003: mediana de los tres últimos intervalos frente a mediana histórica.

## Main leakage risk

Usar mapping supervisado/target de enero como etiquetas históricas, o seguir
seleccionando contra valid. Los controles de cutoff y folds actuales pasan;
persisten riesgos de selección. Aviso NumPy de overflow documentado.

## Recommendation for integrated V2

**Conservar V1.** Revisar guardias e interfaces temporales; ejecutar TR-001
antes de integrar pesos. No comparar 0,47 interno con 0,271 oficial.
Detalles y comandos: [handoff técnico](carles_temporal_recurrence.md).
