"""Contratos del script de comparación de algoritmos (sin entrenar)."""

from __future__ import annotations

import pytest

from run_algorithm_comparison import (
    CLASSIFIER_SPECS,
    DEFAULT_CLASSIFIERS,
    comparison_pipelines,
    parse_args,
    select_tuned,
    select_winner,
    tuning_grid,
)


def test_default_catalog_covers_old_algorithms_and_boosting():
    assert list(DEFAULT_CLASSIFIERS) == list(CLASSIFIER_SPECS)
    assert {
        "naive_bayes",
        "complement_nb",
        "bernoulli_nb",
        "decision_tree",
        "random_forest",
        "knn",
        "mlp",
        "xgboost",
        "catboost",
        "lightgbm",
    } == set(DEFAULT_CLASSIFIERS)


def test_comparison_pipelines_share_representation_and_use_each_estimator():
    pipelines = comparison_pipelines(
        {"method": "count"}, classifiers=["random_forest", "knn"], select_k=10
    )

    assert list(pipelines) == ["random_forest", "knn"]
    assert pipelines["random_forest"].named_steps["classifier"].n_estimators == 1000
    assert pipelines["knn"].named_steps["classifier"].n_neighbors == 50


def test_comparison_pipelines_reject_unknown_names():
    with pytest.raises(ValueError, match="Clasificador no válido"):
        comparison_pipelines({"method": "count"}, classifiers=["not-real"])


def test_select_winner_breaks_ties_by_f1_then_name():
    rows = [
        {"classifier": "b", "accuracy_mean": 0.5, "macro_f1_mean": 0.4},
        {"classifier": "a", "accuracy_mean": 0.5, "macro_f1_mean": 0.4},
        {"classifier": "c", "accuracy_mean": 0.5, "macro_f1_mean": 0.45},
    ]

    assert select_winner(rows)["classifier"] == "c"


def test_select_winner_requires_rows():
    with pytest.raises(ValueError):
        select_winner([])


def test_tuning_grid_starts_with_the_baseline_spec():
    grid = tuning_grid("random_forest")

    assert grid[0] == CLASSIFIER_SPECS["random_forest"]
    assert len({repr(sorted(config.items())) for config in grid}) == len(grid)


def test_tuning_grid_for_alias_reuses_naive_bayes_spec():
    assert tuning_grid("multinomial_nb")[0] == CLASSIFIER_SPECS["naive_bayes"]


def test_select_tuned_prefers_higher_accuracy():
    rows = [
        {"classifier_params": '{"a": 1}', "accuracy_mean": 0.4, "macro_f1_mean": 0.4},
        {"classifier_params": '{"a": 2}', "accuracy_mean": 0.6, "macro_f1_mean": 0.5},
    ]

    assert select_tuned(rows)["classifier_params"] == '{"a": 2}'


def test_cli_maps_zero_select_k_to_none_and_parses_classifier_list():
    args = parse_args(["--select-k", "0", "--classifiers", "random_forest,xgboost"])

    assert args.select_k is None
    assert args.classifiers == ["random_forest", "xgboost"]


def test_cli_all_selects_the_whole_catalog():
    assert parse_args(["--classifiers", "all"]).classifiers is None
