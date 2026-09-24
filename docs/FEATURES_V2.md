# Features V2 candidatas para UBS

`src/transaction_forecasting/features/v2.py` genera una fila numérica por
`client_id` a partir del historial anterior al corte. No usa etiquetas ni
modifica el pipeline. La salida comparte índice con la tabla de
`ClientFeatureBuilder` de `origin/main:src/transaction_forecasting/ubs/features.py`
y sus columnas tienen prefijo `v2_` para permitir un `join` sin colisiones.

```python
from transaction_forecasting.features.v2 import build_client_v2_features

extra = build_client_v2_features(
    train_transactions,
    train_client_ids,
    cutoff="2026-01-01",
    groups=("recency", "frequency", "intervals", "amount", "description"),
)
# train_features = existing_train_features.join(extra)
```

La llamada rechaza cualquier transacción en el corte o posterior, incluso de
otro cliente, y admite clientes sin transacciones para la inferencia. Respeta el
orden de `client_ids` recibido. Nunca pasar la columna de target a la función.

| Grupo | Candidatas principales | Precaución |
| --- | --- | --- |
| `recency` | Recuento, historial escaso, días desde último evento, amplitud del historial | Valores ausentes para cliente sin historial |
| `frequency` | Eventos en ventanas 7/14/30/60/90/180 días, cuota de últimos 30 días | Las ventanas se anclan al corte, no a la última transacción |
| `intervals` | Mediana y desviación de gaps, cuota semanal y mensual | Intervalos cero se excluyen del cálculo de cadencia |
| `amount` | Mediana, MAD, CV, cambio de los últimos 3 importes | Solo moneda del evento más reciente; no se convierten divisas |
| `description` | Concentración, entropía, número de descripciones repetidas; streams semanales/mensuales y de importe estable | Los streams se separan por descripción y moneda; contrastar en validation |

Los grupos se pueden activar individualmente con `groups=(...)`. Son hipótesis
para un experimento sobre exactamente los mismos clientes y Macro-F1 oficial de
V1. No hay puntuaciones de mejora: faltan los archivos brutos en este checkout.
El análisis publicado en `origin/main:docs/UBS_DATASET_ANALYSIS.md` motiva las
features de streams, pero no sustituye la evaluación controlada. Las features
de descripción no codifican asociaciones con clases; cualquier estadística
supervisada adicional deberá ajustarse exclusivamente en train.
