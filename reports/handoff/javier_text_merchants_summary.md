# Javier — texto, descripciones y merchants

**Mejor resultado nuevo:** blend fijo 75% V1 + 25% merchant proxy, Macro-F1
**0.243527**, accuracy **0.300**. Delta frente a V1 (**0.2710243**):
**-0.027497**. La V1 sigue siendo el mejor predictor.

**Tres hallazgos:**

1. Texto word 1–2 mejora F1 de `none` de 0.0942 a 0.3891, pero baja las
   siete familias y obtiene Macro-F1 0.200045.
2. El blend con merchant proxy mejora `none` a 0.3969 y `cloud` a 0.3143;
   empeora las otras seis familias y no supera la V1.
3. No hay merchant ID real. La diversidad de descripciones aumenta entre
   train y validación (mediana ≈24–26 frente a 30–36 por cliente), lo que
   vuelve frágiles las frecuencias aprendidas.

**Tres recomendaciones:** conservar la V1; probar texto por secuencia
recurrente con validación por clientes en train; mantener vocabularios y
asociaciones ajustados sólo con train y excluir la contribución del propio
cliente. No integrar los clasificadores ni blends actuales como predictor.

**Commits:** `f598fc3` contiene utilidades, runner, tests y resultados;
`7ef31a6` añade blends fijos y tracking. Revisarlos en ese orden si se
necesita infraestructura experimental.

**Riesgo principal:** descripciones sintéticas con palabras literales de la
familia y cambio de distribución entre train y validación. Para V2, evaluar
una única ablación de texto local a la secuencia recurrente contra la V1.

Informe completo: `reports/handoff/javier_text_merchants.md`.
