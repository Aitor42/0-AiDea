"""Escritura del CSV de predicciones de Kaggle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nli_project.data.dataset import VALID_PREDICTION_LABELS


def write_submission(ids, predictions, path: str | Path) -> pd.DataFrame:
    """Genera el CSV de submission de Kaggle con el esquema oficial.

    Args:
        ids: Identificadores de las filas de test.
        predictions: Etiquetas predichas; deben pertenecer a {0, 1, 2}.
        path: Ruta del CSV de salida.

    Returns:
        DataFrame con las columnas ``id`` y ``prediction``.

    Raises:
        ValueError: Si la longitud de ids y predicciones no coincide o si
            alguna predicción está fuera de {0, 1, 2}.
    """
    ids_list = list(ids)
    predictions_array = np.asarray(predictions)

    if len(ids_list) != len(predictions_array):
        raise ValueError(
            "Desajuste de longitudes en la submission: "
            f"{len(ids_list)} ids frente a {len(predictions_array)} predicciones."
        )

    values = predictions_array.tolist()
    invalid = [value for value in values if value not in VALID_PREDICTION_LABELS]
    if invalid:
        raise ValueError(
            "Las predicciones deben pertenecer a {0, 1, 2}; "
            f"valor no válido: {invalid[0]!r}"
        )

    submission = pd.DataFrame(
        {"id": ids_list, "prediction": [int(value) for value in values]}
    )
    submission_path = Path(path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    return submission
