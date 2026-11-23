"""Serialización de informes del pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def save_results(rows, path: str | Path) -> Path:
    """Escribe o añade filas de resultados en un CSV, sin duplicar cabecera.

    Si el fichero ya existe, las filas nuevas se añaden tras las existentes y
    las columnas se unifican (uniendo las de ambas versiones); las celdas sin
    dato quedan vacías (NaN al leer). No se escriben cabeceras duplicadas.

    Args:
        rows: Secuencia de diccionarios con los resultados.
        path: Ruta del CSV de resultados.

    Returns:
        La ruta del fichero escrito.
    """
    frame = pd.DataFrame(list(rows))
    results_path = Path(path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if results_path.exists():
        existing = pd.read_csv(results_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(results_path, index=False)
    return results_path



def _json_native(value):
    """Convierte tipos numpy/pandas a tipos nativos serializables a JSON.

    Los flotantes no finitos (NaN/inf) se convierten en ``None`` porque el
    JSON estándar no los admite.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return _json_native(value.tolist())
    if isinstance(value, pd.Series):
        return _json_native(value.tolist())
    if isinstance(value, pd.DataFrame):
        return _json_native(value.to_dict(orient="records"))
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_native(item) for item in value]
    return str(value)



def save_json(obj, path: str | Path) -> Path:
    """Serializa ``obj`` a JSON aceptando tipos numpy y pandas.

    Args:
        obj: Objeto a serializar (diccionarios, listas, numpy, pandas...).
        path: Ruta del fichero JSON.

    Returns:
        La ruta del fichero escrito.
    """
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_native(obj), handle, ensure_ascii=False, indent=2)
    return json_path
