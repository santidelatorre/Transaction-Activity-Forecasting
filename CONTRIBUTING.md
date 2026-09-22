# Contribuir

Gracias por colaborar en el proyecto. Mantenemos `main` estable y trabajamos mediante Pull Requests.

## Flujo diario

Use the repository setup script for a reproducible local environment. Before opening a Pull Request, run the tests, Ruff checks, and `pre-commit run --all-files`. Keep notebooks focused and clean unnecessary outputs before committing; reusable logic belongs in `src/`.

Antes de empezar:

```bash
git checkout main
git pull origin main
git checkout -b feature/nombre-de-la-tarea
```

Usa una rama por tarea. Prefijos recomendados: `feature/`, `experiment/`, `fix/`, `refactor/` y `docs/`. Ejemplos: `feature/recurrence-detection`, `experiment/catboost-baseline`.

Antes de abrir el Pull Request:

```bash
ruff check .
ruff format --check .
pytest
```

Después, crea un commit claro, sube la rama y abre un Pull Request hacia `main`:

```bash
git add .
git commit -m "Descripción clara"
git push -u origin nombre-de-la-rama
```

Pide al menos una revisión, mantén los cambios enfocados y evita commits gigantes. No subas datasets, credenciales, `.env` ni artefactos grandes. La lógica reutilizable debe vivir en `src/`, no dentro de notebooks.

Actualiza tu rama con frecuencia para reducir conflictos:

```bash
git checkout main
git pull origin main
```

Tras el merge, vuelve a actualizar `main` antes de comenzar la siguiente tarea.
