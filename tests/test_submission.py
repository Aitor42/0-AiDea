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



@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[:-1],
        lambda rows: rows + [rows[-1]],
        lambda rows: rows[:2] + [rows[1]] + rows[3:],
        lambda rows: [rows[0], rows[2], rows[1], *rows[3:]],
        lambda rows: [rows[0], "id-0,", *rows[2:]],
        lambda rows: [rows[0], "id-0,3", *rows[2:]],
        lambda rows: ["id,label", *rows[1:]],
    ],
    ids=["missing-row", "extra-row", "duplicate-id", "swapped-ids", "empty-label", "bad-label", "bad-header"],
)
def test_validate_submission_rejects_invalid_csv(tmp_path, mutate):
    from minilm_submission import TEST_ROWS, validate_submission

    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    write_submission([f"id-{i}" for i in range(TEST_ROWS)], [0] * TEST_ROWS, sample)
    validate_submission(sample, sample)

    rows = sample.read_text(encoding="utf-8").splitlines()
    submission.write_text("\n".join(mutate(rows)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_submission(submission, sample)
