# Contribuir

## Rama personal

Cada participante trabaja en `dev/<nombre>`; Carles utiliza `dev/carles`.
Mantener `main` estable. No subir cambios directamente a `main`.

Si tu rama ya existe localmente:

```bash
git switch dev/carles
git fetch origin
git merge origin/main
```

Si todav?a no existe ni localmente ni en el remoto, crearla una sola vez:

```bash
git fetch origin
git switch -c dev/carles origin/main
```

Sustituir `carles` por tu nombre. Si ya existe en GitHub, usar
`git switch --track origin/dev/carles` para obtenerla localmente.
El merge de `origin/main` se hace estando en tu rama personal; no modifica main.
Resolver conflictos y ejecutar pruebas antes de subir cambios.

## Cambios y revisi?n

Usar el script de setup del repositorio. Mantener notebooks enfocados y mover
la l?gica reutilizable a `src/`. Antes de proponer cambios:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m pre_commit run --all-files
```

Revisar `git status` y a?adir ?nicamente los archivos de la aportaci?n:

```bash
git add ruta/al/archivo
git commit -m "Descripci?n clara"
git push -u origin dev/carles
```

Abrir un PR desde la rama personal hacia `main`. Abrir el PR no realiza el merge.
Pedir al menos una revisi?n. La integraci?n la ejecuta la persona autorizada por
el equipo despu?s de revisar los cambios y las comprobaciones.

No subir datasets, credenciales, `.env`, directorios temporales ni artefactos
grandes. Mantener commits acotados incluso cuando varias tareas comparten rama.
Despu?s de un merge aprobado, actualizar la rama personal mediante
`git fetch origin` y `git merge origin/main`.
