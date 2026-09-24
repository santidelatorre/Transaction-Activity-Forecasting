# Propuesta de producto y aportación de Carles

Revisión del 24 de septiembre de 2026. Propuesta para discutir con el equipo;
no asigna tareas ni modifica main. Se han contrastado ideas del equipo con el
enunciado, sin incorporar la transcripción privada al repositorio.

Fuentes públicas:

- Contrato UBS: [versión y detalle](OFFICIAL_CHALLENGE.md).
- Evento: https://ai-weeks.ch/2026/hack-zurich (24–25 septiembre, 36 horas;
  posicionamiento público de IA ética, fiable y abierta).
- Listado de challenges: https://ai-weeks.ch/2026/challenges.

La información visible del evento no establece aquí una rúbrica verificable de
premios ni permisos específicos de datos/APIs. El enunciado UBS sí concreta la
puntuación predictiva. No convertir el marketing general del evento en reglas.

## Qué rescatamos de la conversación

| Idea | Decisión propuesta | Motivo y prueba necesaria |
|---|---|---|
| Probar modelos y justificar la elección | Prioridad alta | Mismo train/valid, macro-F1 de ocho clases y errores por familia; registrar tiempo y versión |
| Dashboard para un gestor o cliente | Sí, pequeño y conectado al predictor | Mostrar siguiente familia, evidencia histórica y límites; usuario y caso de uso se acuerdan con mentor |
| Usuario elige regresión, árboles o Transformer | Solo vista técnica opcional | El producto elige el modelo; comparar algoritmos es trabajo interno |
| Modelos distintos para perfiles distintos | Experimento posterior | Un modelo global primero; segmentación solo si mejora valid sin sobreajustar grupos pequeños |
| Perfiles de gasto | Sí, como descriptores observables | Recencia, frecuencia, regularidad, mezcla de categorías y longitud de historia; verificar utilidad con ablación |
| Perfiles por edad, género o región | No sustentado actualmente | No hay esos campos en el esquema publicado; no inferirlos de compras |
| Predecir aceptación de hipoteca, crédito o pensión | Fuera del target actual | No hay outcomes de ofertas/aceptación; necesitaría otro dataset y otra evaluación |
| Publicidad o venta de datos | No usar como argumento principal | No aparece como objetivo del reto ni se ha demostrado autorización o valor con estos datos |
| Detectar fraude | Posible extensión, no capacidad demostrada | El texto lo menciona como motivación; no hay etiquetas de fraude para validar un detector |

No deducimos rasgos psicológicos como “impulsivo” de un patrón de compras.
Podemos medir variabilidad o regularidad sin atribuir causas personales.
La transcripción contiene hipótesis legales/comerciales: no se consideran hechos
ni autorizaciones. No hace falta resolver esas hipótesis para cumplir el reto.

## Producto recomendado

Un asistente de próximos compromisos recurrentes: ayuda a anticipar una familia
de gasto y entender la evidencia que la respalda. Tiene utilidad directa para
recordatorios, revisión de suscripciones y planificación. Estas utilidades
aparecen en la motivación de UBS; retención o ahorro operativo serían hipótesis
de negocio por validar, no beneficios medidos en este dataset.

Demo mínima para un cliente:

1. Historial visible hasta el corte.
2. Familia predicha, incluyendo la posibilidad `none`.
3. Ejemplos de transacciones y señales realmente utilizadas.
4. Aviso claro si fecha/importe son estimaciones auxiliares.
5. Acción ilustrativa: revisar suscripciones o preparar un recordatorio.

No mostrar “probabilidad 90%” a partir de un score heurístico. Si el modelo
produce probabilidades, evaluar su calibración antes de presentarlas como tal.
Tampoco afirmar que falta saldo: el esquema publicado no incluye saldo inicial.
Una suma de flujos observados no equivale al saldo disponible.

Ejemplo de pitch: “Predecimos la próxima familia de gasto recurrente para que el
cliente pueda anticipar sus compromisos. Comparamos alternativas con macro-F1
y mostramos evidencia histórica y limitaciones junto a la predicción.” Añadir
solo resultados efectivamente medidos; no sustituirlos por métricas del mock.

## Experimentos, por orden

1. **Control simple:** clase mayoritaria aprendida solo de train. Mide el nivel
   trivial y prueba la entrega; no se espera que gane macro-F1.
2. **Clasificador de texto por cliente:** agregar descripciones previas al corte,
   TF-IDF ajustado con train y regresión logística multiclase. La regresión
   logística clasifica; la regresión lineal ordinaria no es el objetivo adecuado
   para etiquetas como gym/insurance/none.
3. **Features temporales y tabulares:** contar apariciones, recencia, dispersión
   de intervalos, calendario mensual y variabilidad de importes por moneda y
   grupo de descripciones; probar un único modelo de árboles.
4. **Texto + recurrencia:** comprobar si la regularidad y el vencimiento probable
   ayudan a distinguir “históricamente frecuente” de “próximo en ocurrir”.
5. **Si hay margen:** combinar modelos con errores complementarios. Predefinir
   pocas variantes; seleccionar reiteradamente sobre valid puede sobreajustar.

Ideas diferenciadoras que sí se pueden comprobar:

- **Calendario frente a días fijos:** mensual no significa cada 30 días. Comparar
  ambos features manteniendo el resto constante.
- **Próximo frente a más frecuente:** analizar los clientes con varias familias.
  El siguiente evento puede pertenecer a una familia menos frecuente.
- **Cancelación frente a pausa:** recencia relativa al período histórico como
  señal para `none`, sin convertir baja confianza en `none` automáticamente.
- **Sugerir revisión en casos ambiguos:** cuando dos modelos discrepen, mostrar
  evidencia en la demo; el CSV sigue necesitando una única etiqueta válida.
- **Pruebas de robustez:** historial truncado, referencias numéricas o variación
  de descripciones, conservando el corte. Reportar cambios en predicciones como
  estabilidad; no llamarlos accuracy nueva si los targets dejan de ser válidos.

## Comparación inicial de repositorios (histórica)

Esta tabla describe la revisión inicial de `d67605d`, no el estado actual.
Después se incorporó `origin/main` en `1a4b336`, que ya incluye adaptador UBS,
features, modelos y runner de entrenamiento. Ver la actualización más abajo.

| Componente | Main en d67605d | Preparación local previa | Decisión inicial |
|---|---|---|---|
| Organización | Paquete modular, CI, pandas y tests | Paquete distinto `taf`, stdlib | Mantener `transaction_forecasting`; no duplicar dos pipelines |
| Predicción | Comercio más frecuente; fecha por mediana de intervalos | Ranking de streams mock, calendario y overdue | Portar señales más adelante, después de inspeccionar los datos oficiales |
| Evaluación | Smoke cuenta filas/clientes; no mide aciertos | Ranking, cobertura y errores de fecha/importe mock | Añadir macro-F1 oficial y cobertura estricta por cliente |
| Validación temporal | Split por filas; puede separar timestamps iguales | Límites temporales y tests de invariancia | Corregir empate temporal; usar train/valid oficial para elegir modelos |
| Confianza | Valores heurísticos 0.6/1.0 | Score etiquetado como heurístico | No presentar el campo legacy como probabilidad |
| Datos | Contrato mock exige merchant y transaction_id | Contrato mock diferente y verdad latente separada | Adaptador oficial pendiente; no renombrar description a familia sin resolverla |
| Seguimiento | Sin resultados predictivos registrados | Manifiestos, hashes, config y métricas por run | Reutilizar el diseño de registro cuando exista el runner oficial |
| Documentación | Supuestos todavía provisionales | Handbook técnico amplio, también basado en mock | Añadir contrato oficial compacto; handbook solo como referencia formativa |

Otros límites del baseline legacy: selecciona por frecuencia, no por próximo
vencimiento; una sola observación recibe un intervalo de 30 días; no recibe un
corte explícito ni modela `none`. Mantenerlo para smoke, no presentarlo como
baseline oficial ya validado. No modificar su interfaz durante este cambio.

## Aportación preparada en dev/carles

- Evaluador independiente de los modelos: ocho clases fijas, alineación por ID,
  macro-F1, métricas por clase y confusión.
- Comprobador de entrega: columnas, etiquetas y conjunto exacto de clientes.
- Tests manualmente comprobables: incluir `none`, no perder clientes y no
  depender del orden de filas.
- Corrección del split temporal para timestamps empatados y datos sin fecha.
- Contrato oficial y esta propuesta para que las otras ramas compartan objetivos.

No se incorporan datasets, transcripción, respuestas del formulario, credenciales
ni resultados mock como evidencia oficial. No se añade un dashboard, modelo
entrenado, adaptador de datos ni entrega competitiva en esta aportación.

El adaptador y los clasificadores ya llegaron con UBS V1 desde main. La interfaz
común sigue siendo `client_id,predicted_next_recurring_merchant`; las siguientes
aportaciones deben ampliar ese recorrido y evitar crear un segundo adaptador.

## Actualización al integrar main 1a4b336 en dev/carles

- Se conservan `temporal_split` con empates agrupados y
  `train_validation_test_split` con sus contratos y grupos de tests originales.
  El helper de tres particiones mantiene su semántica de filas y embargo; no
  se cambia el entrenamiento UBS para usar ninguno de estos splits genéricos.
- `evaluation.official.classification_metrics` centraliza las métricas de ocho
  clases; `ubs.evaluation.evaluate_predictions` adapta al formato que ya consume
  el runner (incluidos `f1-score`, matriz y distribución).
- Vocabulario, corte y nombres de columnas se comparten entre ambos módulos.
- `ubs.data.validate_submission` delega esquema, clases y cobertura al núcleo
  compartido, manteniendo orden estricto, comprobación de clientes de test y
  retorno None. El helper CSV acepta filas desordenadas y devuelve copia alineada.
- Tests de compatibilidad verifican valores manuales, clases ausentes, errores
  de cobertura, orden y ausencia de mutación. No se eliminan los tests UBS V1.
- No se incluye el script local de emergencia ni datasets en este merge.

## Verificación inicial (antes de integrar UBS V1)

- Python 3.12.10, pandas 2.3.3, NumPy 2.3.5, pytest 8.4.2, Ruff 0.6.9.
- `python -m pytest -q`: 20 tests correctos, incluidos los tests anteriores.
  Windows impidió escribir una caché de pytest; no afectó a las pruebas.
- `python -m ruff check .` y `python -m ruff format --check .`: correctos.
- El smoke anterior sigue ejecutándose: 7 transacciones, 2 clientes.
- CLI `score` y `validate` comprobados con CSV pequeños de resultado conocido;
  invocación del módulo por terminal comprobada.
- Ninguna puntuación sobre el dataset oficial ni rendimiento de modelos
  entrenados se ha medido en esta aportación.

El entorno local de comprobación contiene las dependencias necesarias para
estos módulos, no todo el conjunto opcional de herramientas de modelado del
proyecto. No se ha cambiado `pyproject.toml`.
