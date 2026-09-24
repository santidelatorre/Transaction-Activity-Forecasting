# EDA: recurrencia, calendario y texto

## Evidencia disponible (24-09-2026)

No hay archivos UBS crudos en este checkout para recalcular cifras. Sin embargo,
`origin/main:docs/UBS_DATASET_ANALYSIS.md` contiene un análisis reproducible
generado con `scripts/analyze_ubs_dataset.py` sobre datos locales ignorados por
Git. Las cifras siguientes proceden de ese informe y **no se han revalidado
contra los archivos originales en esta rama**. El `mock_transactions()` de esta
rama sigue siendo sintético y no debe usarse como evidencia UBS.

## Hallazgos medidos en el informe de `origin/main`

- En train hay 33.051 grupos repetidos `(client_id, description)` y los 2.000
  clientes tienen alguna descripción repetida. La mediana de apariciones de un
  grupo repetido es 3. Solo el 30,69 % de esos grupos cumple desviación estándar
  de intervalos `<= 3` días; el 25,22 % tiene CV de importe `<= 5 %`. Hay
  recurrencia abundante, pero frecuencia de repetición y regularidad son
  propiedades diferentes.
- Las bandas aproximadas de periodicidad de train clasifican 13.819 grupos
  como trimestrales, 9.338 mensuales, 7.802 anuales, 1.286 quincenales y
  796 semanales. Son bins descriptivos del informe, no series verificadas ni
  porcentajes de clientes. La ventana histórica de unos 13 meses hace que
  interpretaciones anuales o trimestrales necesiten especial cautela.
- `description` tiene 1.475 valores únicos en train. Los términos más comunes
  incluyen `shop` (13.187), `salary` (12.031), `plan` (10.922), `service` (8.880)
  y `atm` (8.083). Esos recuentos son de términos en transacciones, no de
  clientes, y su frecuencia global no implica poder discriminativo.
- El texto histórico sí muestra asociaciones con las ocho clases: `cloud access`
  aparece en el 80,53 % de clientes `cloud` frente al 15,91 % del resto;
  `phone contract` en 82,72 % de `mobile` frente a 17,47 %; `member pass` en
  76,26 % de `music` frente a 16,54 %; y `member plan` en 47,24 % de `none`
  frente a 7,48 %. Son asociaciones de **train**, no rendimiento validado.
  Además, la palabra literal `mobile` o `music` aparece en 0 % de los clientes
  de esas clases, de modo que buscar solo el nombre exacto de la etiqueta
  perdería ambas señales.
- Descripciones con aparente pureza, como `member plan` (72,87 % `none` entre
  387 clientes) o `digital service` (70,88 % `none` entre 419), pueden ser
  atajos propios del generador. Deben comprobarse en validación con clientes
  distintos antes de incorporarse como reglas o codificaciones supervisadas.

Sí hay un hallazgo verificable sobre el código, distinto de un hallazgo del
dataset: `models.baseline.predict_next()` agrupa por el texto literal de
`merchant`, elige el comercio más frecuente y predice con la mediana de días
entre sus eventos. Si solo hay un evento, usa 30 días; considera regular un
historial con al menos dos intervalos y desviación estándar de hasta tres días.
Esto sirve como punto de comparación, pero no reconoce variantes del nombre,
meses de duración variable, cobros quincenales anclados al calendario ni varias
series recurrentes por cliente. No modificarlo hasta observar datos reales y
medir la mejora en un split causal.

## Análisis reproducible al recibir el dataset

Primero confirmar los nombres reales de cliente, fecha, descripción/comercio,
importe, moneda, dirección y clase, así como el instante de predicción. Formar
un historial por cliente usando solo eventos anteriores a ese instante.
Conservar dos vistas de texto: literal y normalizada. La normalización inicial
puede plegar mayúsculas, espacios, puntuación y números variables, pero debe
conservar una copia del valor original para auditar colisiones. No fusionar
contrapartes distintas solo por compartir tokens genéricos.

| Pregunta | Medida y desglose | Control necesario |
| --- | --- | --- |
| ¿Cuántas series se repiten? | Por cliente y descripción/comercio, proporción con al menos 2, 3 y 4 eventos; clientes distintos por serie; cobertura de transacciones | Denominador de clientes y de series por separado; excluir duplicados exactos |
| ¿Qué cadencias aparecen? | Histograma de intervalos consecutivos; masa cerca de 1, 7, 14 y 28–31 días; mediana, MAD y p90 de cada serie | Separar eventos con mismo timestamp; no interpretar intervalo cero como recurrencia |
| ¿Hay anclaje de calendario? | Cuota en mismo día de semana, día de mes, último día hábil o fin de mes; errores respecto a fecha semanal/mensual esperada | Los meses tienen 28–31 días; festivos y fines de semana desplazan cargos |
| ¿Qué texto domina? | Top descripciones literales y normalizadas por transacciones y clientes únicos; cobertura top 10/50; tasa de fragmentación de variantes | Evitar publicar nombres de personas, cuentas o texto sensible |
| ¿Cuántas series tiene cada cliente? | Número de comercios repetidos, cuota de eventos del más frecuente y entropía de comercio | Un top por volumen puede ocultar otra serie predictiva |
| ¿Cómo difieren las ocho clases? | Repetir las medidas anteriores por clase: soporte de clientes, mediana/IQR y prevalencia con intervalo de confianza | Calcular solo en train; mostrar tasas dentro de cada clase y comparación con el total |
| ¿Qué ocurre con poco historial? | Separar clientes con 0, 1, 2 y 3+ transacciones antes del corte; soporte por clase y disponibilidad de cada estadístico | Intervalos y estabilidad no existen con 0–1 eventos; documentar los nulos |

Para cada señal por clase, registrar número de clientes, porcentaje dentro de
la clase y diferencia frente al resto con la misma ventana de observación.
Contrastar sobre todo clientes con longitud de historial comparable: una clase
puede parecer más periódica solo por contar con más meses observados. Repetir
el análisis por cortes temporales; publicar únicamente agregados cuando el
texto pueda identificar a alguien.

## Features candidatas para probar, no para activar todavía

Todas se calculan hasta `as_of` excluido. Los estadísticos de una serie requieren
una clave de comercio/descripción estable; si falta o cambia mucho, usar una
versión a nivel cliente y marcar la cobertura. Calcular por moneda y dirección
cuando el importe o la recurrencia dependan de ellas.

| Feature | Definición operativa | Soporte mínimo |
| --- | --- | --- |
| `series_event_count`, `series_lifespan_days` | Eventos y días entre primero y último de cada cliente–comercio | 1, 2 respectivamente |
| `series_recency_days` | Días desde el último evento anterior a `as_of` | 1 |
| `gap_median_days`, `gap_mad_days` | Mediana y desviación absoluta mediana de intervalos positivos ordenados | 2 y 3 eventos |
| `gap_last_over_median` | Último intervalo dividido por la mediana, con división protegida | 3 eventos |
| `weekly_phase_error` | Mediana de distancia en días al múltiplo de 7 más próximo, sobre intervalos positivos | 3 eventos |
| `monthly_calendar_error` | Error en días entre cada evento y el siguiente esperado al sumar un mes calendario al anterior; comparar anclaje a fin de mes | 3 eventos |
| `weekday_mode_share`, `monthday_mode_share`, `month_end_share` | Fracción de eventos en día modal de semana/mes y al cierre de mes | 2 eventos; suavizar con soporte |
| `description_repeat_share`, `description_unique_count` | Cuota de eventos en descripciones que reaparecen y número de descripciones distintas | 1 |
| `merchant_top_share`, `merchant_entropy` | Concentración de eventos por comercio normalizado | 1 |
| `text_variant_count` | Número de textos literales asociados al mismo comercio normalizado | 1; auditar fusiones |
| `series_count_30d`, `series_count_90d` | Eventos de la serie en ventanas previas de 30/90 días | 0 |

Las señales de periodicidad son continuas; no convertirlas enseguida en reglas
binarias de «semanal» o «mensual». Mantener valores nulos y banderas de soporte
para series cortas. Contrastar familias de features por separado en el mismo
split y con la Macro-F1 oficial, una vez confirmados target y métrica. Un token
o comercio con asociación aparente a una clase debe evaluarse con soporte de
clientes únicos y fuera de muestra: las frecuencias globales o codificaciones
por target aprendidas sobre validación causarían leakage.
