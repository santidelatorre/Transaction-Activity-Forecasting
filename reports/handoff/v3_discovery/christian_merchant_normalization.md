# Christian — merchant normalization discovery

Fecha: 2026-09-24

Rama: `discovery/christian-merchant-normalization`

Base: `5884ddd07a19b2e19b44b4fd55afdf4eb6dcf749`

## Conclusión

**EXACT DESCRIPTION IS ALREADY SUFFICIENT**

La normalización conservadora encuentra aliases plausibles, pero el efecto sobre los streams de
TRAIN es pequeño y no se traduce en una mejora predictiva. El ensemble reduce 302 de 50.720 streams
(-0,60 %), eleva la media de 2,9073 a 2,9247 eventos y añade 18 candidatos. En la comparación fija
con V2, el Macro-F1 oficial baja de **0,391549** a **0,385680** (-0,005870). No se recomienda integrar
esta normalización en V2.

## Protocolo sin leakage

- El normalizador se aprendió únicamente con `train_transactions.jsonl` y
  `unlabeled_pretrain_transactions.jsonl`: 897.394 transacciones, 12.000 clientes y 2.673
  descriptions distintas.
- No se usaron labels de validation, ni validation para formar clusters o escoger umbrales.
- La configuración y el mapping se escribieron antes de abrir validation.
- Validation se cargó una vez en la ejecución congelada para comparar exact y normalized con el
  mismo scorer y con la misma receta V2.
- Test no se cargó ni se utilizó.
- La evaluación interna usa un split estratificado fijo 75/25 de clientes de TRAIN. El mapping
  no supervisado puede usar todo TRAIN, pero el mapa description→clase del scorer se ajusta solo en
  los 1.500 clientes internos de fit.

## Normalización evaluada

Se compararon exact match, lowercase, limpieza de whitespace/punctuation, token sorting,
eliminación de tokens genéricos, normalización ligera de abreviaturas/sufijos, similitud de char
n-grams, token Jaccard, TF-IDF y un ensemble conservador.

El ensemble solo acepta un merge cuando hay un token informativo compartido y al menos dos señales
léxicas. También exige compatibilidad ≥0,82, ausencia de conflictos estables de MCC/type/direction,
distancia logarítmica de amount ≤0,35 y distancia relativa de cadencia ≤0,35. El clustering usa
complete-link y limita cada grupo a 20 descriptions para impedir cadenas de similitud. Palabras como
`plan`, `digital`, `premium`, `plus` o `service` nunca bastan por sí solas.

| Método | Grupos en TRAIN | Merges | Streams | Media eventos | Streams ≥3 | Candidatos | Cobertura candidatos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact | 1.475 | 0 | 50.720 | 2,9073 | 46,778 % | 1.256 | 46,75 % |
| Token sorting | 1.457 | 18 | 50.694 | 2,9088 | 46,814 % | 1.257 | 46,80 % |
| Generic tokens | 371 | 1.104 | 47.263 | 3,1200 | 51,827 % | 1.530 | 54,05 % |
| Light stemming | 1.324 | 151 | 50.459 | 2,9224 | 47,155 % | 1.277 | 47,25 % |
| Char n-gram | 1.259 | 216 | 50.550 | 2,9171 | 46,983 % | 1.261 | 46,90 % |
| Token Jaccard | 1.232 | 243 | 50.573 | 2,9158 | 46,972 % | 1.264 | 46,90 % |
| TF-IDF | 1.358 | 117 | 50.710 | 2,9079 | 46,790 % | 1.256 | 46,75 % |
| Ensemble conservador | 1.117 | 358 | 50.418 | 2,9247 | 47,178 % | 1.274 | 47,25 % |

Lowercase y whitespace/punctuation son idénticos a exact: el dataset ya viene limpio para esas
transformaciones. La eliminación directa de genéricos parece fuerte en métricas de tamaño, pero
fusiona 1.104 de 1.475 descriptions y reduce la estabilidad MCC/type; se considera demasiado
agresiva y no se usa como resultado final.

## Calidad intrínseca: exact frente a ensemble

| Métrica | Exact | Ensemble | Cambio |
| --- | ---: | ---: | ---: |
| Streams | 50.720 | 50.418 | -302 (-0,60 %) |
| Eventos medios por stream | 2,9073 | 2,9247 | +0,0174 |
| Streams con ≥2 eventos | 65,164 % | 65,683 % | +0,519 pp |
| Streams con ≥3 eventos | 46,778 % | 47,178 % | +0,399 pp |
| Streams con ≥4 eventos | 30,913 % | 31,185 % | +0,272 pp |
| MAD mediana de gaps | 22,206 días | 22,182 días | -0,025 días |
| MAD relativa mediana | 0,45719 | 0,45761 | +0,00042 (peor) |
| Amount CV mediana | 0,37740 | 0,37668 | -0,00072 |
| Periodicidad interpretable | 6,841 % | 6,932 % | +0,091 pp |
| MCC/type estable | 75,538 % | 75,336 % | -0,202 pp |
| Streams candidatos | 1.256 | 1.274 | +18 |
| Clientes cubiertos | 935 (46,75 %) | 945 (47,25 %) | +10 (+0,50 pp) |

La mejora de cantidad es consistente pero pequeña. La regularidad y el amount apenas cambian. La
caída de estabilidad MCC/type es una señal de que algunos merges añaden ruido aunque pasen los
guardrails.

## Evaluación predictiva

El scorer mínimo usa exactamente la misma regla, pesos y umbrales en ambos brazos. En el split
interno ambos producen Macro-F1 0,057396 y predicen `none` para los 500 clientes. En validation el
delta es +0,000150, también con una predicción casi degenerada en `none`. Por ello este scorer sirve
como control de invariancia, pero no aporta evidencia útil a favor de la normalización y no se
ajustaron sus umbrales después de observar validation.

La prueba útil es la integración mínima con la receta V2 congelada. Solo cambia la description que
identifica los streams; arquitectura, pesos del blend y parámetros quedan fijos.

| Clase | F1 V2 exact | F1 V2 normalized | Delta |
| --- | ---: | ---: | ---: |
| cloud | 0,45000 | 0,45783 | +0,00783 |
| gym | 0,48727 | 0,48375 | -0,00352 |
| insurance | 0,42915 | 0,42353 | -0,00562 |
| mobile | 0,45267 | 0,44444 | -0,00823 |
| music | 0,16993 | 0,16149 | -0,00844 |
| none | 0,51938 | 0,50794 | -0,01144 |
| software | 0,37398 | 0,38710 | +0,01311 |
| streaming | 0,25000 | 0,21935 | -0,03065 |
| **Macro-F1** | **0,391549** | **0,385680** | **-0,005870** |

Accuracy también baja de 0,424 a 0,417. La mejora de dos clases no compensa el deterioro de las
otras seis, especialmente `streaming`.

## Merges buenos

Ejemplos aceptados con MCC/type coherentes y alta compatibilidad:

- `cloud access`, `cloud access core`, `cloud access plus`, `pay cloud access` → `cloud access`;
- `media streaming`, `media streaming service` → `media streaming`;
- `phone contract`, `phone contract digital`, `pay phone contract` → `phone contract`;
- `electronics shop`, `electronics shop core`, `pay electronics shop`,
  `pay electronics shop core` → `electronics shop`;
- `cover plan`, `cover plan core` → `cover plan`.

Estos casos muestran que los aliases existen. Su frecuencia dentro de un mismo cliente es demasiado
baja para cambiar materialmente la calidad global de los streams.

## Merges peligrosos

Las señales léxicas propusieron pares que los guardrails rechazaron o dejaron para revisión:

- `digital plus` ↔ `premium plan digital plus`: comparten tokens genéricos, pero chocan en
  type/direction y amount;
- `member member plan` ↔ `member plan`: Jaccard 1,0, pero chocan en type/direction y la distancia de
  amount es 1,46;
- `plan service` ↔ `service plan`: token sorting/Jaccard sugieren igualdad, pero amount y cadencia
  difieren ampliamente;
- `software access` ↔ `software access plus`: similitud léxica alta, pero distancia de cadencia 0,43;
- `video access` ↔ `video access plus`: similitud léxica alta, pero distancia de cadencia 0,46.

También requieren revisión los clusters grandes y raros formados por permutaciones repetidas de
tokens genéricos. Son precisamente el motivo para favorecer precisión de merge y no adoptar la
variante `generic_tokens`.

## Artefactos

- Código: `scripts/experiments/christian_merchant_normalization.py`
- Configuración congelada: `frozen_configuration.json`
- Mapping aprendido: `description_mapping.csv`
- Métricas intrínsecas: `intrinsic_comparison.csv`
- Comparación interna: `internal_predictive_comparison.json`
- Comparación oficial del scorer: `official_recurrence_comparison.json`
- Comparación V2: `official_v2_comparison.json`
- Ejemplos: `good_merges.csv` y `dangerous_merges.csv`

Todos los outputs están en
`outputs/metrics/v3_discovery/christian_merchant_normalization/`. Los datasets permanecen locales e
ignorados por Git. V2 no fue modificado.

## Reproducción

```bash
python scripts/experiments/christian_merchant_normalization.py
```

La ejecución es determinista (`seed=42`) y escribe el mapping/configuración antes de cargar
validation.
