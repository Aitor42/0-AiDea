"""Contratos de validación cruzada, evaluación holdout y persistencia."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nli_project.training.training import (
    build_pipeline,
    cross_validate,
)
from nli_project.evaluation.metrics import (
    evaluate_holdout,
    per_language_metrics,
)
from nli_project.evaluation.reporting import (
    save_json,
    save_results,
)


def _pipeline():
    return build_pipeline(
        {"method": "tfidf", "max_features": 5_000, "min_df": 1},
        classifier_params={"n_estimators": 20, "min_samples_leaf": 1},
        use_pos_features=False,
        use_relation_features=False,
        select_k=50,
    )


def test_cross_validate_returns_fold_table_summary_and_confusion(synthetic_pairs):
    result = cross_validate(
        synthetic_pairs,
        _pipeline(),
        n_splits=3,
        seed=42,
        group_by_premise=False,
    )

    folds = result["folds"]
    assert isinstance(folds, pd.DataFrame)
    assert len(folds) == 3
    assert {
        "fold",
        "n_train",
        "n_test",
        "accuracy",
        "macro_f1",
        "fit_seconds",
        "predict_seconds",
    } <= set(folds.columns)
    assert (folds["accuracy"].between(0, 1)).all()
    assert (folds["macro_f1"].between(0, 1)).all()
    assert (folds["fit_seconds"] >= 0).all()
    assert (folds["predict_seconds"] >= 0).all()

    summary = result["summary"]
    assert summary["n_splits"] == 3
    assert summary["accuracy_mean"] == pytest.approx(folds["accuracy"].mean())
    assert summary["accuracy_std"] == pytest.approx(folds["accuracy"].std(ddof=0))
    assert summary["macro_f1_mean"] == pytest.approx(folds["macro_f1"].mean())
    assert summary["fit_seconds_total"] == pytest.approx(folds["fit_seconds"].sum())

    confusion = result["confusion_matrix"]
    assert confusion.shape == (3, 3)
    assert confusion.sum() == len(synthetic_pairs)


def test_cross_validate_is_reproducible(synthetic_pairs):
    first = cross_validate(synthetic_pairs, _pipeline(), n_splits=3, seed=42)
    second = cross_validate(synthetic_pairs, _pipeline(), n_splits=3, seed=42)

    # El contrato de reproducibilidad cubre las salidas deterministas (folds,
    # métricas y matriz de confusión). Los tiempos se miden con el reloj y se
    # cuantizan a 0.1 s: pausas del recolector de basura de ~0.1 s pueden
    # mover la cuantización entre dos ejecuciones idénticas, de modo que se
    # compara solo su no negatividad y no su igualdad exacta.
    deterministic_columns = [
        column
        for column in first["folds"].columns
        if column not in ("fit_seconds", "predict_seconds")
    ]
    pd.testing.assert_frame_equal(
        first["folds"][deterministic_columns], second["folds"][deterministic_columns]
    )
    assert (second["folds"]["fit_seconds"] >= 0).all()
    assert (second["folds"]["predict_seconds"] >= 0).all()
    np.testing.assert_array_equal(
        first["confusion_matrix"], second["confusion_matrix"]
    )


def test_cross_validate_grouped_folds_do_not_share_premises(synthetic_pairs):
    groups_seen = []
    result = cross_validate(
        synthetic_pairs,
        _pipeline(),
        n_splits=3,
        seed=42,
        group_by_premise=True,
        return_fold_frames=True,
    )

    for fold_train, fold_test in result["fold_frames"]:
        assert set(fold_train["premise"]).isdisjoint(set(fold_test["premise"]))
        groups_seen.append(set(fold_test["premise"]))

    assert len(groups_seen) == 3
    assert not groups_seen[0] & groups_seen[1]


def test_cross_validate_does_not_mutate_input(synthetic_pairs):
    before = synthetic_pairs.copy(deep=True)
    cross_validate(synthetic_pairs, _pipeline(), n_splits=3, seed=42)
    pd.testing.assert_frame_equal(before, synthetic_pairs)


def test_evaluate_holdout_reports_competition_and_per_class_metrics(synthetic_pairs):
    train_df, val_df = synthetic_pairs.iloc[:120], synthetic_pairs.iloc[120:]

    result = evaluate_holdout(_pipeline(), train_df, val_df)

    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["macro_f1"] <= 1
    assert result["n_train"] == len(train_df)
    assert result["n_val"] == len(val_df)
    assert result["runtime_seconds"] >= 0
    assert result["confusion_matrix"].shape == (3, 3)
    assert result["confusion_matrix"].sum() == len(val_df)

    report = result["classification_report"]
    for label_name in ("entailment", "neutral", "contradiction"):
        assert label_name in report
        assert {"precision", "recall", "f1-score", "support"} <= set(report[label_name])

    assert isinstance(result["per_language"], pd.DataFrame)
    assert {"language", "n", "accuracy", "macro_f1"} <= set(
        result["per_language"].columns
    )
    assert result["per_language"]["n"].sum() <= len(val_df)


def test_evaluate_holdout_does_not_fit_on_validation(synthetic_pairs):
    train_df = synthetic_pairs.iloc[:300]
    val_df = synthetic_pairs.iloc[300:].copy()
    val_df["premise_processed"] = "heldoutonlytoken"
    val_df["hypothesis_processed"] = "heldoutonlytoken"

    pipeline = _pipeline()
    evaluate_holdout(pipeline, train_df, val_df)

    vocabulary = pipeline.named_steps["features"].vectorizer_.vocabulary_
    assert "heldoutonlytoken" not in vocabulary


def test_per_language_metrics_filters_small_groups():
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 2, 1, 0, 0])
    languages = np.array(["en"] * 8 + ["es"] * 2)

    table = per_language_metrics(y_true, y_pred, languages, min_samples=3)

    assert list(table["language"]) == ["en"]
    assert table.iloc[0]["n"] == 8
    assert 0 <= table.iloc[0]["accuracy"] <= 1
    assert 0 <= table.iloc[0]["macro_f1"] <= 1
    assert table.iloc[0]["n_classes"] == 3


def test_save_results_writes_csv_with_union_of_columns(tmp_path):
    path = tmp_path / "results.csv"
    rows = [
        {"experiment": "a", "accuracy": 0.5},
        {"experiment": "b", "accuracy": 0.6, "macro_f1": 0.55},
    ]

    save_results(rows, path)

    stored = pd.read_csv(path)
    assert list(stored["experiment"]) == ["a", "b"]
    assert {"experiment", "accuracy", "macro_f1"} <= set(stored.columns)
    assert stored.loc[0, "macro_f1"] != stored.loc[0, "macro_f1"]  # NaN


def test_save_results_appends_without_duplicating_header(tmp_path):
    path = tmp_path / "results.csv"
    save_results([{"experiment": "a", "accuracy": 0.5}], path)
    save_results([{"experiment": "b", "accuracy": 0.7}], path)

    stored = pd.read_csv(path)
    assert list(stored["experiment"]) == ["a", "b"]


def test_save_json_serializes_numpy_types(tmp_path):
    path = tmp_path / "result.json"
    payload = {
        "accuracy": np.float64(0.75),
        "confusion": np.array([[1, 2], [3, 4]]),
        "labels": np.array([0, 1, 2]),
    }

    save_json(payload, path)

    stored = json.loads(path.read_text())
    assert stored["accuracy"] == 0.75
    assert stored["confusion"] == [[1, 2], [3, 4]]
    assert stored["labels"] == [0, 1, 2]
