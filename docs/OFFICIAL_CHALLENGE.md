# Contrato oficial UBS 2026

Fuente primaria fijada por commit: [UBS Swiss AI Weeks challenge](https://github.com/UBS-AG/Swiss-AI-Weeks/blob/796d5805ec5f8a228a3ec0de36a2b4e6e1b1a1df/hackathons/2026/challenge.md).

Se predice una etiqueta por `client_id`: la próxima familia recurrente después
del corte `2026-01-01`, dentro de un horizonte de 90 días.

- Clases: `cloud`, `gym`, `insurance`, `mobile`, `music`, `software`,
  `streaming`, `none`.
- Target: `target_next_recurring_merchant`.
- Predicción: `predicted_next_recurring_merchant`.
- Métrica: macro-F1 con las ocho clases fijas y F1 cero cuando una clase no tiene
  denominador.
- Submission: exactamente `client_id,predicted_next_recurring_merchant`.

Los ficheros de transacciones usan `client_id`, `timestamp`, `amount`,
`currency`, `direction`, `type`, `mcc`, `description` y `fee`. Todos los features
deben proceder del historial anterior al cutoff. Los IDs sirven para alinear y
agrupar; no son features predictivas.

Los splits oficiales TRAIN, VALID y TEST se mantienen separados por cliente.
TRAIN es la única fuente para fitting y selección de la receta limpia. VALID se
abre solo después del freeze para una evaluación y TEST solo genera la
submission. Los datos sin etiquetas pueden usarse únicamente del modo fijado en
la receta y con checks de solapamiento.

El validator local rechaza columnas, IDs o labels inválidos y reordena solo una
copia según el sample. La evaluación alinea por ID y nunca descarta silenciosamente
clientes sin predicción:

```powershell
python -m ubs_recurrence.official validate --sample data/raw/ubs_2026/sample_submission.csv --predictions outputs/predictions/submission_stream_identity_clean.csv
python scripts/validate_submission.py
```

Los datos oficiales, cachés y modelos se conservan localmente bajo directorios
ignorados. El CSV limpio validado es el único output predictivo principal
versionado.
