# Contrato oficial UBS 2026

Consultado el 24 de septiembre de 2026. Fuente primaria:
https://github.com/UBS-AG/Swiss-AI-Weeks/blob/796d5805ec5f8a228a3ec0de36a2b4e6e1b1a1df/hackathons/2026/challenge.md

Este contrato sustituye las hipótesis previas sobre target, métrica, esquema y
entrega. Los módulos de smoke existentes siguen siendo ejemplos de desarrollo.

## Qué hay que predecir

Una etiqueta por `client_id`: la próxima familia de comercio recurrente después
del corte `2026-01-01`, dentro de un horizonte de 90 días.

- Clases: `cloud`, `gym`, `insurance`, `mobile`, `music`, `software`, `streaming`, `none`.
- Target: `target_next_recurring_merchant`.
- Predicción: `predicted_next_recurring_merchant`.
- Métrica principal: macro-F1, media de los F1 de las ocho clases.
- `none` significa que no se espera recurrencia en ese horizonte. No equivale
  a abstención, error del programa o baja confianza.

Fecha exacta e importe no forman parte del CSV requerido. Pueden ser señales
internas o estimaciones adicionales de la demo, identificadas como tales.

## Datos y límites

El repositorio oficial contiene `hackathons/2026/data/dataset.zip`.
El enunciado enumera transacciones JSONL de train, valid y test; etiquetas CSV
de train y valid; `sample_submission.csv`; y transacciones adicionales sin
etiquetas para preentrenamiento opcional.

Campos de transacción: `client_id`, `timestamp`, `amount`, `currency`,
`direction`, `type`, `mcc`, `description`, `fee`.
No se documentan edad, género, nombre, saldo disponible, aceptación de ofertas,
merchant family por evento ni un ID de transacción nativo.

La descripción es texto observado; no es automáticamente una identidad de
comercio limpia ni una etiqueta de familia. El target por cliente tampoco
etiqueta cada transacción histórica. No sumar monedas distintas sin una
conversión definida. Mantener dirección separada del importe observado.

Conservar originales en `data/raw/`, fuera de Git.

Verificación posterior con el ZIP oficial descargado el 24 de septiembre:
`train_labels.csv` contiene 2.000 filas y `valid_labels.csv` contiene 1.000.
Ambos tienen exactamente el encabezado
`client_id,cutoff_date,target_next_recurring_merchant`; todas las filas tienen
`cutoff_date=2026-01-01`. Por tanto, exigir esa columna en el evaluador es
compatible con los archivos reales. El test de regresión incluye ese encabezado
y terminadores de línea CRCRLF observados en el ZIP, con filas ficticias.

ZIP SHA-256: `1afc95470f4e8641601503172be3e698ef9eaf91528d911a6a01a120911634c6`.
Fuente: https://github.com/UBS-AG/Swiss-AI-Weeks/blob/main/hackathons/2026/data/dataset.zip
Esta comprobación del esquema no sustituye la auditoría completa de transacciones.

## Evaluación que corresponde a este reto

Entrenar con train y medir/modelar decisiones con valid. Test solo se usa para
producir predicciones; sus etiquetas no están disponibles. Comprobar el
solapamiento de clientes entre archivos al auditar los datos. El ID sirve para
unir tablas, no debe entrar como característica por defecto.

Protocolo local reforzado para Milestone 2: reservar un holdout estratificado de
clientes dentro de train para todas las decisiones de configuración, calibración
y ensemble. Ajustar preprocessing y mappings solo en los clientes de fit;
reajustar la receta elegida con train completo, sin valid. El runner puntúa
valid oficial una vez, después de congelar la elección, y distingue ese score
del de selección interna. Esto no revierte la exposición de valid en runs
anteriores. Es una decisión experimental local, no una nueva regla de UBS.

Todos los features proceden del historial disponible al corte. Ajustar
vocabularios, imputaciones, escalados y otros componentes aprendidos con train,
sin usar las etiquetas de valid. El enunciado permite historias sin etiquetas
para preentrenamiento: documentar cualquier uso y compararlo en una ablación.

No reemplazar el split oficial por un corte arbitrario de filas transaccionales.
Una separación temporal adicional sirve como diagnóstico si se pueden construir
targets correctos en otros cortes. No reutilizar el target de enero para un
corte histórico diferente. `evaluation.temporal.temporal_split` es un helper
genérico, no el evaluador oficial completo.

Para cada clase, F1 = 2 TP / (2 TP + FP + FN). La media da el mismo peso a cada
familia y a `none`; accuracy puede ocultar errores en clases poco frecuentes.
El helper local usa las ocho clases fijas y F1=0 cuando el denominador es cero.
La convención para clases ausentes debe confirmarse con el evaluador del
organizador si se proporciona; queda explícita en la salida local.

## Comandos disponibles

El flujo de entrenamiento oficial ya está implementado en
`scripts/run_ubs_baseline.py --config configs/ubs_v1.toml`, con adaptador en
`ubs.data` y features/modelos en `ubs/`. Los comandos siguientes son utilidades
para comprobar archivos; comparten métricas y controles con el runner UBS V1.
Este conserva su formato de informe y su exigencia de orden idéntico al sample.

Con el paquete instalado, o `PYTHONPATH=src`, evaluar un CSV de validación:

```powershell
python -m transaction_forecasting.evaluation.official score --labels data/raw/valid_labels.csv --predictions outputs/predictions/valid.csv
```

Comprobar un CSV de test:

```powershell
python -m transaction_forecasting.evaluation.official validate --sample data/raw/sample_submission.csv --predictions outputs/predictions/submission.csv
```

Estos comandos leen archivos y no envían ni sobrescriben nada. `score` devuelve
macro-F1, accuracy, precision/recall/F1/support por clase y matriz de confusión.
Ambos rechazan clientes repetidos, ausentes o extra, IDs vacíos y etiquetas
inválidas. La evaluación alinea por ID, no por posición, y nunca elimina
silenciosamente clientes sin predicción. La entrega admite exactamente las dos
columnas del ejemplo y su orden. `validate_submission()` devuelve una copia
ordenada según el sample, si otro módulo necesita escribirla.

## Entrega y hitos

```csv
client_id,predicted_next_recurring_merchant
C000004,none
C000008,none
```

Usar todos y solo los clientes del sample, una vez cada uno. El formulario de
entrega está enlazado en el enunciado y solicita nombre del equipo, repositorio
y CSV. No se ha realizado ninguna entrega desde este cambio.

Hitos publicados, hora CEST de Zúrich: día 1 antes de 12:00 y 17:00; día 2 antes
de 12:00 y 17:30. Dentro de cada hito cuenta la última entrega válida anterior
al cierre. La clasificación final usa el mejor macro-F1 de las entregas válidas
de los hitos. No asumir que la última entrega es siempre mejor: revisar valid
y conservar el CSV de cada versión. Confirmar cualquier actualización presencial.

## Preguntas que siguen abiertas

- Inclusión exacta del instante/día de corte y del extremo del horizonte.
- Cómo se define recurrencia en la generación de etiquetas y cómo se desempatan
  eventos simultáneos. Las etiquetas suministradas siguen siendo la referencia.
- Reglas específicas sobre APIs externas y código preparado; la existencia de
  créditos o de un dataset sintético no responde por sí sola a estas preguntas.
- Criterios y ponderaciones del jurado general, separados del ranking macro-F1
  del challenge. La página pública consultada no permite confirmar una rúbrica.
