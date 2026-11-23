"""Localiza el proyecto desde el directorio de trabajo de un notebook."""

from pathlib import Path


def repository_root() -> Path:
    directory = Path.cwd().resolve()
    for candidate in (directory, *directory.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/nli_project").is_dir():
            return candidate
    raise RuntimeError("Abre el notebook desde la raíz del repositorio o una de sus subcarpetas.")
