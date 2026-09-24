# Guía de agentes para el hackathon

Estas instrucciones complementan, sin sustituir, las normas de `README.md`,
`CONTRIBUTING.md`, `docs/` y la configuración del repositorio. En caso de
conflicto, se debe preservar la seguridad, la reproducibilidad, las interfaces
compartidas y las instrucciones más específicas del área afectada.

## Objetivo operativo

El equipo dispone de crédito limitado de API. Priorizar, en este orden:

1. Resolver correctamente.
2. Mantener el ritmo de desarrollo y la integración de la demo.
3. Evitar consumo innecesario de modelos o razonamiento costosos.

Usar el modelo y el nivel de razonamiento menos costosos que resuelvan la tarea
con fiabilidad. No elegir opciones más caras «por si acaso».

## Selección de modelo y razonamiento

### GPT-6 Luna

Preferir para trabajo de bajo riesgo, claro, localizado y fácil de comprobar:

- renombrados, formato, limpieza y documentación;
- comentarios, README y boilerplate sencillo;
- tests triviales y scripts pequeños;
- extracción o transformación simple de datos;
- cambios mecánicos repetitivos y revisión superficial de estilo.

Usar razonamiento `low` por defecto; `medium` solo para coordinar varios cambios
sencillos.

### GPT-6 Sol

Es el modelo de referencia para el desarrollo normal y para cualquier tarea que
no encaje claramente en Luna o Astra. Preferirlo para features, backend,
frontend, APIs, pipelines de datos, integración, análisis de datos, ML
convencional, debugging no trivial, refactors, diseño de módulos, tests
importantes, Git, revisión de PRs y decisiones técnicas habituales.

Usar razonamiento `medium` por defecto; `low` para problemas sencillos y `high`
solo cuando exista complejidad real.

### GPT-6 Astra

Es una escalada excepcional, no una opción por defecto. Reservarla para
decisiones de arquitectura críticas, bugs difíciles tras intentos razonables con
Sol, análisis complejo de causa raíz, componentes críticos, problemas
matemáticos o algorítmicos complejos, evaluación avanzada de ML, trade-offs de
alto impacto, integración final de alto riesgo y revisión crítica de la demo.

Empezar con razonamiento `low` o `medium`; usar `high` únicamente cuando la
dificultad lo justifique. No escalar a Astra porque una tarea sea larga: debe ser
intelectualmente difícil o de alto riesgo.

## Estrategia de coste

Antes de escalar de Luna a Sol o de Sol a Astra:

- mejorar el contexto disponible e inspeccionar solo los archivos relevantes;
- buscar en el repositorio y ejecutar pruebas, linters o código para verificar;
- dividir el problema en cambios pequeños y comprobables;
- evitar repetir llamadas costosas o reanalizar grandes áreas sin necesidad.

Si la plataforma permite seleccionar o delegar modelos, aplicar esta política
sin pedir confirmación para cada decisión. Si no permite cambiar el modelo de
forma autónoma, continuar con el disponible; solo ante una ventaja importante,
indicar: `Recomiendo escalar esta tarea a [modelo] con reasoning [nivel] porque
[motivo].`

## Prioridades del hackathon

- Favorecer cambios pequeños, verificables e integrables rápidamente.
- Mantener interfaces entre componentes estables y detectar pronto bloqueos
  entre los siete miembros del equipo.
- Ejecutar las pruebas pertinentes tras cambios relevantes y hacer commits
  claros y acotados.
- Evitar refactors innecesarios y no romper código funcional por mejoras
  cosméticas.
- Para decisiones difíciles, priorizar el impacto sobre la demo y la entrega.

## Verificación y seguridad

- Inspeccionar el código existente antes de modificar arquitectura.
- Ejecutar los tests relevantes y, cuando corresponda, `ruff check .`,
  `ruff format --check .` y `pre-commit run --all-files`.
- Comprobar imports y respetar `.gitignore`.
- Nunca añadir `OPENAI_API_KEY`, otras claves API, credenciales, `.env`, datasets
  ni artefactos grandes al repositorio.
- No realizar cambios destructivos sin necesidad.

## Colaboración existente

Seguir el flujo de ramas y Pull Requests definido en `CONTRIBUTING.md` y el
plan de workstreams de `docs/HACKATHON_PLAN.md`: mantener `main` estable, usar
la rama personal correspondiente, mantener los notebooks enfocados y mover la
lógica reutilizable a `src/`.

## Política de ramas del equipo

- Cada participante trabaja en su rama personal `dev/<nombre>`, con seguimiento
  de la rama remota del mismo nombre. Por ejemplo, Carles usa `dev/carles` y
  Santiago usa `dev/santiago`.
- Nunca hacer commits, pushes ni merges directos a `main`. La integración en
  `main` se realiza exclusivamente mediante Pull Request desde una rama de
  trabajo publicada.
- Antes de una tarea importante, comprobar la rama actual, actualizar las
  referencias remotas y evaluar si se deben incorporar los cambios recientes de
  `origin/main` a la rama personal. No sobrescribir trabajo local o remoto.
- Los commits y pushes ordinarios se realizan en la rama personal, con cambios
  pequeños y mensajes descriptivos. Una rama puede contener varias tareas, pero
  cada commit y Pull Request debe mantener un alcance claro y revisable.
- Antes de un commit relevante, ejecutar las comprobaciones disponibles:
  pre-commit, Ruff, formato, tests e imports cuando correspondan.
- Ante conflictos al incorporar cambios de `main`, conservar en lo posible la
  intención de ambos lados y no descartar cambios automáticamente.
- Mantener fuera del control de versiones los archivos personales, temporales,
  credenciales, entornos y cachés mediante patrones apropiados de `.gitignore`.
  `.gitignore` protege archivos locales; las ramas aíslan el trabajo de cada
  miembro del equipo.
