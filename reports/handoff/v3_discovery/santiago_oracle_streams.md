# Oracle de streams y reconstrucción del target

## Resumen ejecutivo

**Conclusión: MIXED EVIDENCE.** Los streams exactos contienen la familia correcta para
459 de 707 clientes positivos de validation (candidate-family recall **0.649222**) y
permiten un oracle de Macro-F1 **0.769375**, frente a **0.391549** de V2. Sin embargo,
el detector produce **13.124** streams y **3.355** familias candidatas por cliente de
media; 450 de los 459 aciertos de cobertura tienen varias familias candidatas. La
cobertura es alta para `streaming` (0.835052) pero baja para `music` (0.430108) y
`gym` (0.421488). Además, 289/293 clientes `none` tienen al menos un stream due con
familia asignada y son indistinguibles de los positivos con este detector.

La evidencia justifica desarrollar streams como generador de candidatos y fuente de
features, pero no justifica todavía reemplazar V2 por una arquitectura central
exclusivamente basada en streams exactos.

## Procedencia e hipótesis

- Commit base congelado: `5884ddd07a19b2e19b44b4fd55afdf4eb6dcf749`.
- Rama: `discovery/santiago-oracle-streams`.
- Cutoff: `2026-01-01T00:00:00Z`; horizonte cerrado: 90 días.
- V2 se mantuvo intacta. Referencia reproducida existente: Macro-F1
  **0.3915494559**, 75% CatBoost de 146 features históricas + 25% periodicidad.
- Hipótesis: el target se puede descomponer como cliente → streams recurrentes →
  próxima fecha → familia → siguiente stream dentro del horizonte.

El experimento mide el techo de esa descomposición. No construye ni selecciona V3.

## Metodología

### Definición exacta de stream y candidato

Un stream es la clave exacta **`(client_id, description)`**, sin normalización adicional.
Se ordenan sus eventos previos al cutoff y se calculan:

- apariciones, timestamps, último y penúltimo timestamp;
- vector de gaps, mediana, media robusta winsorizada al 10%, desviación y MAD;
- recencia, media/mediana/desviación/MAD/CV del importe;
- moda de MCC, `type`, `direction` y `currency`;
- periodicidad aproximada (`weekly`, `biweekly`, `monthly`, `quarterly`, `annual`,
  `other`) y próxima fecha proyectada.

La definición se mantuvo deliberadamente sencilla y liberal para medir un techo:
stream recurrente = al menos dos apariciones. La proyección usa la mediana del gap y,
si la siguiente repetición queda antes del cutoff, avanza ciclos completos hasta la
primera fecha en o después del cutoff. Es candidato si esa fecha cae como máximo 90
días después del cutoff. No hay threshold de regularidad o estabilidad de importe.

### Mapping description → family

El mapping se ajusta exclusivamente con TRAIN y solo sobre streams candidatos. Para
cada descripción calcula, por familia positiva, el lift suavizado de presencia entre
clientes de esa clase frente al resto. Conserva una única familia si el soporte es al
menos dos clientes y el lift ganador es al menos 1.5. Los valores se fijaron antes de
leer labels de validation. El mapping final contiene 136 descripciones elegibles y
asigna familia al 33.8616% de los streams candidatos de validation.

Como diagnóstico de transferencia interna —no como ajuste— se ejecutaron cinco folds
estratificados y disjuntos por `client_id` dentro de TRAIN. El recall OOF fue 0.792587
y el Macro-F1 del oracle con abstención fue 0.874428. La caída a 0.649222/0.769375 en
validation es una advertencia de fragilidad del mapping exacto, no una razón para
retocar thresholds con validation.

### Oracles

- **Oracle A:** para cada positivo comprueba si la familia real pertenece al conjunto
  de familias de sus streams candidatos. Su media es candidate-family recall.
- **Oracle B:** fija como correctos los positivos cubiertos y los `none`; para cada
  positivo no cubierto permite elegir una familia detectada o `none`. Un MILP binario
  asigna esos errores para maximizar exactamente el Macro-F1 oficial de ocho clases.
  La solución óptima coincide aquí con enviar todos los positivos no cubiertos a
  `none`. Es un upper bound diagnóstico y usa labels de validation únicamente para
  puntuar/optimizar el oracle, nunca para producir el detector o mapping.
- **Oracle C:** separa los `none` según ausencia de recurrencia, recurrencia fuera de
  horizonte, candidatos sin familia y candidatos con familia indistinguibles de un
  positivo.

## Controles anti-leakage

1. El loader rechaza cualquier transacción en o después del cutoff.
2. El detector no acepta labels ni usa el valor de `client_id` como feature.
3. El mapper exige que todos sus streams pertenezcan a clientes etiquetados de TRAIN
   y rechaza transformar clientes usados en su ajuste.
4. Los folds OOF del mapping son disjuntos por cliente.
5. El runner construye streams, ejecuta OOF y congela el mapping antes de leer
   `valid_labels.csv`.
6. Validation se usa solo para las métricas finales y para la elección omnisciente del
   oracle; test no se carga ni se usa.
7. No se modifican el evaluador oficial, V2 ni sus outputs.

## Resultados por clase

Candidate-family recall se define solo para clases positivas; para `none` se muestra
0 por no representar una familia candidata. `Oracle F1` es el F1 por clase de la
asignación que maximiza Macro-F1.

| clase | soporte | recall familia candidata | Oracle F1 | V2 F1 | streams cand./cliente | familias cand./cliente |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 89 | 0.640449 | 0.780822 | 0.450000 | 13.270 | 3.640 |
| gym | 121 | 0.421488 | 0.593023 | 0.487273 | 13.256 | 3.355 |
| insurance | 99 | 0.717172 | 0.835294 | 0.429150 | 12.828 | 3.141 |
| mobile | 104 | 0.721154 | 0.837989 | 0.452675 | 13.731 | 3.231 |
| music | 93 | 0.430108 | 0.601504 | 0.169935 | 14.172 | 3.688 |
| software | 104 | 0.807692 | 0.893617 | 0.373984 | 14.067 | 3.433 |
| streaming | 97 | 0.835052 | 0.910112 | 0.250000 | 12.969 | 3.196 |
| none | 293 | n/a | 0.702638 | 0.519380 | 12.294 | 3.304 |

Totales:

- candidate-family recall positivos: **459/707 = 0.649222**;
- Oracle Macro-F1: **0.769375**; accuracy: **0.752**;
- V2 Macro-F1: **0.391549**; accuracy: **0.424**;
- media global: **13.124** streams y **3.355** familias candidatas por cliente.

## Breakdown de fallos

De los 707 positivos:

| resultado | clientes | interpretación |
| --- | ---: | --- |
| sin stream recurrente | 0 | la definición liberal siempre encuentra repetición |
| ningún stream temporalmente due | 1 | fallo puro de generación temporal |
| familia mapeada en otro stream, pero no due | 67 | proyección temporal no pone la familia correcta en horizonte |
| miss de mapping/familia | 180 | hay candidatos due, pero ninguno se mapea a la familia correcta |
| familia correcta candidata | 459 | el oracle puede acertar; queda escoger el stream correcto |

Breakdown de aciertos/misses por clase (`candidate`, `mapped-not-due`, `mapping miss`):

| clase | candidate | mapped-not-due | mapping miss | no due |
| --- | ---: | ---: | ---: | ---: |
| cloud | 57 | 7 | 25 | 0 |
| gym | 51 | 14 | 56 | 0 |
| insurance | 71 | 15 | 13 | 0 |
| mobile | 75 | 6 | 22 | 1 |
| music | 40 | 6 | 47 | 0 |
| software | 84 | 8 | 12 | 0 |
| streaming | 81 | 11 | 5 | 0 |

La selección es el cuello de botella después de cobertura: los 459 aciertos tienen
múltiples streams candidatos y 450/459 tienen múltiples familias candidatas; solo 9
tienen una única familia. El oracle no demuestra que un ranker real pueda escogerlas.

### Oracle C: `none`

| categoría `none` | clientes |
| --- | ---: |
| sin stream recurrente convincente | 0 |
| recurrente, pero sin stream due en horizonte | 0 |
| streams due, sin familia asignada | 4 |
| stream due con familia; indistinguible de positivo | 289 |

El detector liberal no resuelve `none`: 98.63% de los `none` tienen evidencia due con
familia. El F1 0.702638 del oracle presupone conocimiento perfecto para abstenerse;
no es alcanzable por la regla temporal sola.

## Music y streaming

`streaming` respalda claramente la hipótesis: recall candidato 0.835052, solo cinco
misses de mapping y Oracle F1 0.910112. Su debilidad en V2 parece compatible con un
problema de ranking/modelo más que con ausencia de candidatos.

`music` muestra lo contrario: recall 0.430108 y 47/93 clientes fallan en el mapping,
a pesar de tener 14.172 candidatos de media. Normalización de descriptions o un mapper
de familia más robusto son requisitos previos; un ranker no puede recuperar la familia
si no entra al conjunto candidato.

## Limitaciones

- Dos eventos bastan y no se exige regularidad: es apropiado para recall/techo, pero
  infla candidatos y vuelve especialmente optimista el oracle.
- Las labels son por cliente, no por evento. El mapping aprende asociaciones débiles
  entre todos los candidatos due y un único target futuro; no identifica el evento
  causal.
- La description es exacta. No se evalúan alias, typos ni normalización semántica.
- La mediana de gaps ignora meses de longitud variable, cancelaciones, cambios de fase
  y mixtures de cadencias.
- El oracle usa verdad de validation para elegir y abstenerse. Solo mide posibilidad;
  no es una estimación de rendimiento desplegable.
- V2 y esta auditoría comparten la validation oficial, ya reutilizada previamente; la
  comparación es diagnóstica, no una estimación independiente de generalización.

## Conclusión

Los criterios predeclarados exigen para evidencia fuerte recall total ≥0.75, Oracle
Macro-F1 ≥0.65 y recall ≥0.60 tanto en `music` como en `streaming`; evidencia débil se
declara si recall total u Oracle Macro-F1 es <0.45. El experimento obtiene 0.649222,
0.769375, 0.430108 y 0.835052 respectivamente: supera el criterio de techo pero falla
los criterios de cobertura total y `music`, sin caer en evidencia débil.

**MIXED EVIDENCE**

Recomendación: conservar V2 y usar streams como generador/features en el siguiente
experimento. Priorizar mapping de familia train-only más robusto, proyección temporal
calendar-aware y un modelo explícito de `none`; después medir si un ranker puede
resolver la ambigüedad de los 13.124 candidatos medios sin sacrificar Macro-F1.

## Reproducción

```powershell
python scripts/experiments/santiago_oracle_streams.py
```

Los artefactos locales se guardan en
`outputs/metrics/v3_discovery/santiago_oracle_streams/` (ignorado por Git):
`summary.json`, `validation_streams.csv`, `validation_client_audit.csv`,
`train_family_mapping.csv` y `class_breakdown.csv`.
