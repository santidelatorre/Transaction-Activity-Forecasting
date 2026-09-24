# Javi: EDA de recurrencia UBS

## Datos y reproducción

Análisis ejecutado el 24-09-2026 sobre el ZIP público de
[`UBS-AG/Swiss-AI-Weeks`](https://github.com/UBS-AG/Swiss-AI-Weeks/blob/main/hackathons/2026/data/dataset.zip),
SHA-256 `1afc95470f4e8641601503172be3e698ef9eaf91528d911a6a01a120911634c6`.
Los archivos extraídos están en `data/raw/ubs_2026/`, ignorados por Git. El corte
es `2026-01-01` UTC y las etiquetas son **por cliente**, no por transacción.

```powershell
python scripts/analyze_javi_recurrence.py --data-dir data/raw/ubs_2026
```

El script usa solo la biblioteca estándar de Python y guarda los agregados en
`outputs/metrics/javi_recurrence.json` (también ignorado por Git). Comprueba que
los clientes etiquetados coinciden con los historiales y que no hay eventos desde
el corte. Train y validation se analizan por separado; no se modificó el pipeline
ni se usaron etiquetas de validation para construir features.

## Periodicidad e intervalos

Una *serie* es un par `(client_id, description)` con al menos dos eventos. Las
bandas siguientes clasifican la **mediana** de sus intervalos: semanal hasta 10
días, quincenal hasta 18, mensual hasta 45, trimestral hasta 110 y anual hasta
400. Son categorías descriptivas; «mensual» aquí no prueba un cargo anclado al
calendario.

| Banda | Train, todas | Validation, todas | Train, regular con ≥3 eventos | Validation, regular con ≥3 eventos |
| --- | ---: | ---: | ---: | ---: |
| Semanal | 796 | 414 | 27 | 14 |
| Quincenal | 1.286 | 554 | 27 | 13 |
| Mensual | 9.338 | 3.714 | 492 | 86 |
| Trimestral | 13.819 | 6.769 | 203 | 106 |
| Anual | 7.802 | 5.034 | 68 | 32 |
| Otros | 10 | 4 | 0 | 0 |
| **Total** | **33.051** | **16.489** | **817** | **251** |

«Regular» exige **al menos tres eventos** y desviación estándar de los intervalos
≤3 días. En train, 9.325 de las 33.051 series repetidas tienen solo **un**
intervalo; en validation son 6.276 de 16.489. Un solo intervalo siempre produce
desviación cero. El informe general anterior comunicaba 30,69 % de grupos con
desviación ≤3 días en train y 39,58 % en validation incluyendo esos casos. Con
soporte mínimo, quedan 817/33.051 (2,47 %) y 251/16.489 (1,52 %).
Por eso conviene almacenar por separado número de eventos, número de intervalos
y dispersión; un indicador binario de «regular» sin soporte exagera la evidencia.

## Familias frente a `none`

`any_family` agrupa las siete clases distintas de `none`. Las tasas de recurrencia
son porcentajes de **clientes** con al menos una serie que cumple el criterio
anterior. «Mensual saliente» exige además que todos los eventos de esa serie
tengan `direction=out`.

| Partición y etiqueta | Clientes | Mediana de transacciones | Clientes con serie regular | Clientes con serie mensual saliente regular | Último evento >30 días antes del corte |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train, `any_family` | 1.403 | 75 | 31,36 % | 18,03 % | 1 |
| Train, `none` | 597 | 64 | 38,69 % | 21,44 % | 9 |
| Validation, `any_family` | 707 | 75 | 23,20 % | 5,66 % | 0 |
| Validation, `none` | 293 | 65 | 20,14 % | 5,80 % | 3 |

`none` tiene menos transacciones históricas en ambas particiones, pero **también
tiene series repetidas y regulares**. En train su tasa de clientes con alguna
serie regular incluso supera a la de las familias; en validation la relación se
invierte. La regularidad de cualquier descripción no es una regla fiable para
descartar `none`: la clase significa que no se espera una **familia objetivo**
recurrente en los 90 días posteriores, no que el cliente carezca de pagos
repetidos. La inactividad >30 días está concentrada en `none`, pero solo afecta
a 10 clientes train y 3 validation; sirve como hipótesis de feature, no como
umbral decidido con estos pocos casos.

| Clase | Train: clientes con serie regular | Validation: clientes con serie regular |
| --- | ---: | ---: |
| cloud | 33,16 % | 35,96 % |
| gym | 35,79 % | 23,14 % |
| insurance | 32,71 % | 14,14 % |
| mobile | 30,37 % | 24,04 % |
| music | 31,82 % | 25,81 % |
| software | 28,21 % | 24,04 % |
| streaming | 28,00 % | 16,49 % |
| none | 38,69 % | 20,14 % |

Las diferencias entre particiones aconsejan comparar señales **por familia de
descripción**, con recencia respecto a su cadencia y número de observaciones,
antes de convertir «hay recurrencia» en una predicción. Las descripciones son
texto observado; el target por cliente no etiqueta cada transacción histórica.

## Historiales cortos y casos extraños

- El mínimo es 8 transacciones por cliente en train y validation. No hay
  clientes con 0–2 eventos; solo uno tiene ≤10 en train (`none`) y uno en
  validation (`streaming`). La menor amplitud de historial es 177,95 días en
  train y 300,60 en validation. Este dataset **no prueba** un fallback para
  clientes recién incorporados.
- No aparecen pares de transacciones con el mismo timestamp dentro de un cliente
  ni intervalos cero dentro de una serie repetida. Sí hay timestamps iguales
  entre clientes distintos, irrelevantes para esta agrupación.
- Hay 125 series train y 75 validation con algún intervalo superior a 365 días.
  En una ventana histórica de unos 13 meses pueden ser cargos anuales, cambios
  de descripción o coincidencias; no etiquetarlas automáticamente como errores.
- El informe general registra cero duplicados exactos y cero eventos en o tras
  el corte. Los importes están en monedas distintas: cualquier medida de
  estabilidad de importe debe separar monedas.

## Features y comprobaciones propuestas

1. `stream_event_count`, `interval_count`, `gap_median_days`, `gap_std_days` y
   `days_since_last`: conservar `NaN` en dispersión si solo hay un intervalo.
2. Distancia al siguiente vencimiento semanal y mensual **calendario**, además
   de una banda de mediana de días. Comparar ambas definiciones en el mismo split.
3. Número de series salientes regulares y recientes **por familia candidata**,
   usando asociaciones de descripción aprendidas solo en train. Añadir cuota
   de eventos de la serie, estabilidad de importe por moneda y recencia relativa
   a su intervalo típico.
4. Para `none`, probar señales de interrupción: tiempo desde el último evento de
   la serie candidata dividido por su intervalo habitual, ausencia reciente
   pese a recurrencia histórica y longitud del historial. Analizar errores por
   clase y no inferir `none` solo por falta de una serie genérica.
5. Mantener un grupo de análisis para series con dos eventos, gaps >365 días y
   clientes con poca historia. Medir cobertura y Macro-F1 por clase antes de
   activar cualquier nueva regla o feature en el pipeline.

La asociación entre estas medidas y el target es descriptiva; no se ha hecho
un experimento de modelo ni se afirma mejora de Macro-F1.
