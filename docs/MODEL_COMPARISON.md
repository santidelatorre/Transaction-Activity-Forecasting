# Comparación controlada de modelos UBS

`evaluation.model_comparison.compare_classifiers` compara una configuración fija
de Logistic Regression y, cuando están instalados, CatBoost, LightGBM y XGBoost;
también calcula un promedio simple de sus probabilidades si hay al menos dos modelos.
No busca hiperparámetros. Recibe features numéricas ya
construidas por cliente, targets e **IDs explícitos** de train y validación;
rechaza clientes solapados y usa las mismas filas, columnas y orden para todos.
La imputación por mediana se ajusta solo con train. Logistic Regression añade
escalado propio ajustado en train. Cada predicción se devuelve en el orden de
`validation_ids`.

El contrato UBS de `origin/main` define las clases `cloud`, `gym`, `insurance`,
`mobile`, `music`, `software`, `streaming`, `none`; el target es
`target_next_recurring_merchant`. Sus particiones oficiales tienen clientes
disjuntos y un corte en `2026-01-01`. El comparador recibe `LABELS` del módulo
UBS y calcula `sklearn.metrics.f1_score(..., labels=LABELS, average="macro",
zero_division=0)`, incluidas clases ausentes de validación. Es la misma
convención que `ubs.evaluation.evaluate_predictions()` en `origin/main`;
hay que confirmar con las reglas oficiales si esa convención es exactamente la
de la competición.

Uso sobre las features V1 integradas en esta rama:

```python
import pandas as pd

from transaction_forecasting.evaluation.model_comparison import compare_classifiers
from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.features import ClientFeatureBuilder

data = load_ubs_data("data/raw/ubs_2026")
builder = ClientFeatureBuilder().fit(data.train_transactions, data.train_labels)
x_train = builder.transform(data.train_transactions)
x_valid = builder.transform(data.valid_transactions)
y_train = data.train_labels.set_index("client_id")[TARGET_COLUMN].reindex(x_train.index)
y_valid = data.valid_labels.set_index("client_id")[TARGET_COLUMN].reindex(x_valid.index)
result = compare_classifiers(
    pd.concat([x_train, x_valid]),
    pd.concat([y_train, y_valid]),
    train_ids=x_train.index,
    validation_ids=x_valid.index,
    labels=LABELS,
)
print(result.summary)
```

La concatenación de la muestra solo organiza las tablas; el constructor de
features debe continuar ajustándose en train y usando únicamente historiales
anteriores al corte. El módulo no certifica por sí solo la causalidad de los
features de entrada. `client_id` debe permanecer en el índice y nunca entrar
como feature. Si una columna es categórica o texto libre, codificarla dentro de
un transformador ajustado solo con train antes de llamar a este comparador.

No hay datos UBS crudos accesibles en este checkout para ejecutar la comparación ni
resultados que reportar. El análisis y el baseline V1 están integrados en la rama,
pero este módulo queda separado del pipeline y no genera submissions. Una vez
integrado, registrar el hash del split, columnas de features, versiones de
dependencias, semilla y Macro-F1 por clase. Repetir la comparación sobre la
misma validación antes de cualquier tuning; para estimar mejora robusta, usar
después un protocolo adicional definido sobre train, sin consultar test.
