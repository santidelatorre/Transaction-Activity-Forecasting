# CSV de submission

`python -m transaction_forecasting.submission predict` convierte un CSV de predicciones ya generadas en un CSV final ordenado como el fichero de referencia. No ejecuta un modelo. Antes de escribir, comprueba columnas, cardinalidad, IDs únicos, ausencia de valores vacíos, ocho clases permitidas y coincidencia exacta de IDs. La salida tiene solo las dos columnas especificadas, en orden `ID, predicción`. Si falla una comprobación, no escribe la salida.

## Contrato UBS V1

`src/transaction_forecasting/ubs/data.py` define `client_id`, `predicted_next_recurring_merchant` y las clases `cloud gym insurance mobile music software streaming none`. El análisis local de `docs/UBS_DATASET_ANALYSIS.md` registra 1000 clientes de test y una fila por cliente en `sample_submission.csv`. Estos datos describen **UBS V1**; si el reglamento o dataset cambian, hay que usar sus columnas, clases y número de filas reales.

```powershell
python -m transaction_forecasting.submission predict `
  --predictions outputs/predictions/model_predictions.csv `
  --reference data/raw/ubs_2026/sample_submission.csv `
  --output outputs/predictions/submission.csv `
  --id-column client_id `
  --label-column predicted_next_recurring_merchant `
  --classes cloud gym insurance mobile music software streaming none `
  --expected-rows 1000
```

`model_predictions.csv` debe contener exactamente una fila por cliente y solo las columnas `client_id,predicted_next_recurring_merchant`; el orden de sus filas puede ser distinto al de la muestra. `sample_submission.csv` debe contener `client_id` y puede incluir otras columnas. Se aceptan IDs como texto para conservar ceros iniciales. El comando no sobrescribe un CSV existente salvo que se añada `--overwrite`.

El validador UBS V1 de `origin/main` (`ubs.data.validate_submission`) comprueba además la coincidencia con los IDs de `test_transactions`. Este comando independiente solo recibe predicciones y referencia; la verificación cruzada con las transacciones debe ejecutarse cuando se use el loader UBS completo. Ni este documento ni el comando sustituyen la especificación oficial de entrega si esta exige más campos o reglas.
