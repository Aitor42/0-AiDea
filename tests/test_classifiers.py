"""Contratos de la fábrica de clasificadores y del pipeline parametrizado."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from classifiers import (
    CLASSIFIER_NAMES,
    NAIVE_BAYES_NAMES,
    build_classifier,
    is_naive_bayes,
)
from training import build_pipeline


def test_catalog_includes_old_algorithms_and_external_boosting():
    expected = {
        "naive_bayes",
        "multinomial_nb",
        "complement_nb",
        "bernoulli_nb",
        "decision_tree",
        "random_forest",
        "knn",
        "mlp",
        "xgboost",
        "catboost",
        "lightgbm",
    }

    assert expected <= set(CLASSIFIER_NAMES)
    assert len(set(CLASSIFIER_NAMES)) == len(CLASSIFIER_NAMES)


def test_naive_bayes_family_is_recognized():
    assert is_naive_bayes("naive_bayes")
    assert is_naive_bayes("multinomial_nb")
    assert is_naive_bayes("complement_nb")
    assert is_naive_bayes("bernoulli_nb")
    assert not is_naive_bayes("knn")
    assert NAIVE_BAYES_NAMES == {
        "naive_bayes",
        "multinomial_nb",
        "complement_nb",
        "bernoulli_nb",
    }


def test_multinomial_alias_returns_the_same_estimator():
    alias = build_classifier("multinomial_nb", alpha=0.5)
    canonical = build_classifier("naive_bayes", alpha=0.5)

    assert isinstance(alias, MultinomialNB)
    assert type(alias) is type(canonical)
    assert alias.alpha == canonical.alpha == 0.5


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("complement_nb", ComplementNB),
        ("bernoulli_nb", BernoulliNB),
        ("decision_tree", DecisionTreeClassifier),
        ("knn", KNeighborsClassifier),
        ("mlp", MLPClassifier),
    ],
)
def test_factory_returns_the_expected_estimator(name, expected):
    assert isinstance(build_classifier(name), expected)


@pytest.mark.parametrize("name", ["knn", "naive_bayes", "complement_nb", "bernoulli_nb"])
def test_deterministic_estimators_do_not_receive_random_state(name):
    estimator = build_classifier(name, random_state=7)

    assert "random_state" not in estimator.get_params()


@pytest.mark.parametrize(
    "name", ["decision_tree", "random_forest", "mlp", "xgboost", "catboost", "lightgbm"]
)
def test_stochastic_estimators_receive_the_global_seed(name):
    pytest.importorskip(
        {"xgboost": "xgboost", "catboost": "catboost", "lightgbm": "lightgbm"}.get(
            name, "sklearn"
        )
    )
    estimator = build_classifier(name, random_state=7)
    params = estimator.get_params()

    assert params.get("random_state", params.get("random_seed")) == 7


def test_caller_params_have_precedence_over_factory_defaults():
    estimator = build_classifier("random_forest", n_estimators=7, min_samples_leaf=3)

    assert estimator.n_estimators == 7
    assert estimator.min_samples_leaf == 3


def test_mlp_normalizes_hidden_layer_sizes_from_json_list():
    estimator = build_classifier("mlp", hidden_layer_sizes=[256, 128])

    assert estimator.hidden_layer_sizes == (256, 128)


def test_factory_does_not_mutate_input_params():
    params = {"alpha": 2.0}

    build_classifier("naive_bayes", **params)

    assert params == {"alpha": 2.0}


def test_unknown_classifier_is_rejected():
    with pytest.raises(ValueError, match="Clasificador no válido"):
        build_classifier("gradient_boosting_magic")


def test_factory_output_can_be_cloned_by_sklearn():
    estimator = build_classifier("random_forest", n_estimators=5)

    assert clone(estimator).n_estimators == 5


def test_build_pipeline_defaults_to_the_reference_random_forest():
    pipeline = build_pipeline({"method": "count"}, select_k=None)

    assert pipeline.named_steps["classifier"].n_estimators == 1000
    assert pipeline.named_steps["classifier"].max_depth == 50
    assert pipeline.named_steps["classifier"].n_jobs == -1


def test_build_pipeline_selects_naive_bayes_and_forwards_params():
    pipeline = build_pipeline(
        {"method": "count"},
        classifier="naive_bayes",
        classifier_params={"alpha": 3.0, "fit_prior": False},
        select_k=20,
    )
    classifier = pipeline.named_steps["classifier"]

    assert isinstance(classifier, MultinomialNB)
    assert classifier.alpha == 3.0
    assert classifier.fit_prior is False


def test_build_pipeline_selects_knn_without_random_state():
    pipeline = build_pipeline(
        {"method": "tfidf"},
        classifier="knn",
        classifier_params={"n_neighbors": 3, "weights": "distance"},
        select_k=20,
    )
    classifier = pipeline.named_steps["classifier"]

    assert isinstance(classifier, KNeighborsClassifier)
    assert classifier.n_neighbors == 3
    assert "random_state" not in classifier.get_params()


def test_build_pipeline_uses_factory_defaults_when_params_are_none():
    pipeline = build_pipeline({"method": "tfidf"}, classifier="mlp", select_k=20)

    assert isinstance(pipeline.named_steps["classifier"], MLPClassifier)


def test_build_pipeline_rejects_unknown_classifier():
    with pytest.raises(ValueError, match="Clasificador no válido"):
        build_pipeline({"method": "tfidf"}, classifier="not-a-classifier")


def test_naive_bayes_rejects_signed_representations():
    with pytest.raises(ValueError, match="no negativas"):
        build_pipeline(
            {"method": "tfidf_svd", "svd_components": 2},
            classifier="naive_bayes",
            select_k=None,
        )


def test_selection_rejects_signed_representations_early():
    with pytest.raises(ValueError, match="no negativas"):
        build_pipeline({"method": "tfidf_svd", "svd_components": 2})


@pytest.mark.parametrize(
    "name",
    ["naive_bayes", "decision_tree", "knn", "mlp", "xgboost", "catboost", "lightgbm"],
)
def test_every_classifier_fits_and_predicts_on_synthetic_data(name, synthetic_pairs):
    pytest.importorskip(
        {"xgboost": "xgboost", "catboost": "catboost", "lightgbm": "lightgbm"}.get(
            name, "sklearn"
        )
    )
    params = {
        "naive_bayes": {"alpha": 1.0},
        "decision_tree": {"max_depth": 3},
        "knn": {"n_neighbors": 3},
        "mlp": {
            "hidden_layer_sizes": (16,),
            "max_iter": 20,
            "early_stopping": False,
        },
        "xgboost": {"n_estimators": 5, "max_depth": 2},
        "catboost": {"iterations": 5, "depth": 2, "verbose": False},
        "lightgbm": {"n_estimators": 5, "num_leaves": 4, "verbose": -1},
    }[name]

    pipeline = build_pipeline(
        {"method": "tfidf", "min_df": 1},
        classifier=name,
        classifier_params=params,
        select_k=20,
    )
    pipeline.fit(synthetic_pairs, synthetic_pairs["label"])
    predictions = pipeline.predict(synthetic_pairs)

    assert predictions.shape == (len(synthetic_pairs),)
    assert set(np.unique(predictions)) <= {0, 1, 2}
