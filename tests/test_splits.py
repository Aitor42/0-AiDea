"""Contratos de carga de datos y partición train/validación."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from training import (
    LABEL_NAMES,
    load_train_data,
    make_split,
    split_summary,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.csv"


def test_load_train_data_returns_expected_shape_and_columns():
    df = load_train_data(TRAIN_PATH)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12120
    assert list(df.columns) == [
        "id",
        "premise",
        "hypothesis",
        "lang_abv",
        "language",
        "label",
    ]
    assert set(df["label"].unique()) == set(LABEL_NAMES)
    assert df["label"].isna().sum() == 0
    assert df["premise"].isna().sum() == 0
    assert df["hypothesis"].isna().sum() == 0


def test_load_train_data_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_train_data(tmp_path / "does-not-exist.csv")


def test_make_split_is_deterministic_disjoint_and_stratified(synthetic_pairs):
    train_a, val_a = make_split(synthetic_pairs, val_size=0.25, seed=42)
    train_b, val_b = make_split(synthetic_pairs, val_size=0.25, seed=42)

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(val_a, val_b)

    assert len(train_a) + len(val_a) == len(synthetic_pairs)
    assert set(train_a["id"]).isdisjoint(set(val_a["id"]))
    assert (train_a["id"].tolist() + val_a["id"].tolist()).count(train_a["id"].iloc[0]) == 1

    expected_val = round(len(synthetic_pairs) * 0.25)
    assert abs(len(val_a) - expected_val) <= 3

    for label in LABEL_NAMES:
        assert label in set(train_a["label"])
        assert label in set(val_a["label"])


def test_make_split_changes_with_seed(synthetic_pairs):
    _, val_42 = make_split(synthetic_pairs, val_size=0.2, seed=42)
    _, val_7 = make_split(synthetic_pairs, val_size=0.2, seed=7)

    assert set(val_42["id"]) != set(val_7["id"])


def test_grouped_split_keeps_premises_in_a_single_partition(synthetic_pairs):
    train_df, val_df = make_split(
        synthetic_pairs, val_size=0.2, seed=42, group_by_premise=True
    )

    assert len(train_df) + len(val_df) == len(synthetic_pairs)
    assert set(train_df["id"]).isdisjoint(set(val_df["id"]))
    assert set(train_df["premise"]).isdisjoint(set(val_df["premise"]))


def test_split_summary_reports_sizes_classes_languages_and_overlap(synthetic_pairs):
    train_df, val_df = make_split(synthetic_pairs, val_size=0.2, seed=42)
    summary = split_summary(train_df, val_df)

    assert summary["n_train"] == len(train_df)
    assert summary["n_val"] == len(val_df)
    assert set(summary["class_distribution_train"]) == set(LABEL_NAMES)
    assert set(summary["class_distribution_val"]) == set(LABEL_NAMES)
    assert set(summary["language_distribution_val"]) <= set(synthetic_pairs["language"])
    assert summary["premise_overlap"] == len(
        set(train_df["premise"]) & set(val_df["premise"])
    )
    assert isinstance(summary["premise_overlap_ratio"], float)
