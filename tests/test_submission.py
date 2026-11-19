"""Contratos de generación de la submission de Kaggle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import write_submission


def test_write_submission_uses_official_schema(tmp_path):
    ids = [f"id-{i}" for i in range(5195)]
    predictions = np.tile([0, 1, 2], 1732)[:5195]
    path = tmp_path / "submission.csv"

    df = write_submission(ids, predictions, path)

    assert list(df.columns) == ["id", "prediction"]
    assert len(df) == 5195
    assert list(pd.read_csv(path).columns) == ["id", "prediction"]
    assert set(df["prediction"].unique()) <= {0, 1, 2}
    assert list(df["id"]) == ids


def test_write_submission_rejects_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        write_submission(["a", "b"], np.array([0]), tmp_path / "submission.csv")


def test_write_submission_rejects_invalid_labels(tmp_path):
    with pytest.raises(ValueError):
        write_submission(["a", "b"], np.array([0, 7]), tmp_path / "submission.csv")
