# Contribuir

Trabaja en una rama y propone la integración mediante pull request. No hagas
push, merge ni force-push a `main` desde un cambio ordinario.

Antes de commit:

```powershell
pytest
ruff check .
ruff format --check .
pre-commit run --all-files
```

Añade solo los archivos de la tarea. No versiones datasets, credenciales,
entornos, modelos, cachés ni outputs grandes.

Stream Identity clean frozen es la baseline principal. Un cambio en features,
modelos, parámetros, orden de clases, postprocesado o fuentes incluidas en
`configs/stream_identity_clean_frozen.json` requiere una revisión explícita de
la receta. No uses VALID para tomar decisiones nuevas ni presentes un benchmark
manual como test rápido de CI.
