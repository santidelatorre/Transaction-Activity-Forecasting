# Mejor modelo actual — Macro-F1 ≈ 0.3505

## Resumen en una frase

Con las **mismas features V1**, el mejor resultado validado es
**CatBoost pequeño + calibración de probabilidades**, no la heurística V1.

| | V1 baseline | Mejor búsqueda |
|---|---:|---:|
| Modelo | `recurrence_heuristic` | `calibrated_catboost_i300_d4_lr0.05_bal` |
| Macro-F1 valid | **0.2710** | **0.3505** |
| Accuracy valid | 0.266 | 0.340 |
| Delta | — | **+0.0795** |

## Cuál es exactamente el modelo

Nombre interno: `calibrated_catboost_i300_d4_lr0.05_bal`

1. **Base:** CatBoost multiclass
   - iterations = 300
   - depth = 4
   - learning_rate = 0.05
   - class weights balanced
   - features = agregados numéricos V1 (218 columnas; **sin** TF-IDF)

2. **Calibración (sobre valid, después de entrenar en train):**
   - temperature = **2.0**
   - none_bias = **-1.5**
   - Ajusta logits para emitir más `none` y suavizar argmax

3. **Submission:** se reentrena la receta en train+valid y predice test
   - Archivo local: `outputs/predictions/submission_model_search.csv` (gitignored)
   - 1000 filas, schema oficial

## Cómo lo hicimos

1. Congelamos features V1 (`ClientFeatureBuilder`).
2. Probamos ~42 variantes: logistic, CatBoost, LightGBM, XGBoost, ensembles, calibración.
3. **Hallazgo clave:** boosting/logistic *crudos* quedaron **por debajo** de 0.271.
   La subida grande vino de **calibrar** CatBoost depth 4.
4. Runner reproducible:

```bash
python scripts/run_ubs_model_search.py --config configs/ubs_model_search.toml
```

## F1 por clase (mejor modelo)

| class | F1 |
|---|---:|
| cloud | 0.436 |
| gym | 0.404 |
| insurance | 0.384 |
| mobile | 0.372 |
| music | 0.357 |
| software | 0.317 |
| none | 0.283 |
| streaming | 0.250 |

Comparado con V1, `none` sube mucho (~0.09 → ~0.28). Streaming sigue débil.

## Qué NO funcionó

- CatBoost/LightGBM/XGBoost/logistic **sin** calibración (mejor raw ~0.22)
- Ensembles soft-vote sin calibrar (~0.23)
- Árboles más profundos (depth 6) peor que depth 4 tras calibrar

## Código relacionado (ya en la rama)

| path | rol |
|---|---|
| `src/transaction_forecasting/ubs/models.py` | CatBoost + LightGBM + XGBoost + `TemperatureCalibrator` |
| `scripts/run_ubs_model_search.py` | búsqueda + submission |
| `configs/ubs_model_search.toml` | rejilla y rutas |
| `tests/test_ubs_model_search.py` | tests de wrappers |
| commit `358d604` | feat: controlled UBS model search… |

## Riesgo importante (para Carles / integración)

La calibración (`temperature`, `none_bias`) se elige con **valid**, igual que el
`none_bias` de la heurística V1. Hay riesgo de optimismo de selección.
No es leakage de test, pero V2 debería tratarlo como candidato a revisar.

## Relación con el informe de errores V1

- `Reports/santiago_error_analysis.md` diagnostica **por qué falla el baseline 0.271**.
- Este archivo documenta el **candidato mejor (0.3505)** ya corrido en esta rama.
- Recomendación: usar el diagnóstico V1 para features; mantener este CatBoost
  calibrado como receta de modelo a re-evaluar cuando lleguen Features V2.
