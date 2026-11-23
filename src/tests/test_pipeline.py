"""Contratos del Random Forest, la selección de features y el pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion, Pipeline

from nli_project.features.pos_features import POSRatioFeatures
from nli_project.features.relation_features import RelationFeatures
from nli_project.training.training import (
    SELECT_K_FEATURES,
    build_pipeline,
)
from nli_project.features.vectorizer import NLITextVectorizer


def test_build_pipeline_returns_random_forest_over_nli_vectorizer():
    pipeline = build_pipeline(
        {"method": "tfidf"}, use_pos_features=False, use_relation_features=False
    )

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["features", "select", "classifier"]
    assert isinstance(pipeline.named_steps["features"], NLITextVectorizer)
    assert isinstance(pipeline.named_steps["select"], SelectKBest)
    assert pipeline.named_steps["select"].k == SELECT_K_FEATURES
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, RandomForestClassifier)
    assert classifier.n_estimators == 1000
    assert classifier.max_depth == 50
    assert classifier.n_jobs == -1


def test_selection_can_be_disabled():
    pipeline = build_pipeline({"method": "tfidf"}, select_k=None)

    assert list(pipeline.named_steps) == ["features", "classifier"]


@pytest.mark.parametrize("invalid", [0, -5, 2.5, "300", True])
def test_invalid_select_k_is_rejected(invalid):
    with pytest.raises(ValueError, match="select_k"):
        build_pipeline({"method": "tfidf"}, select_k=invalid)


def test_pipeline_defaults_point_to_preprocessed_columns():
    pipeline = build_pipeline(
        {"method": "count"}, use_pos_features=False, use_relation_features=False
    )
    vectorizer = pipeline.named_steps["features"]

    assert tuple(vectorizer.columns) == ("premise_processed", "hypothesis_processed")
    assert vectorizer.method == "count"


def test_vectorizer_params_are_forwarded_without_mutating_input():
    params = {"method": "tfidf_bigram", "max_features": 1234, "use_pos": False}
    pipeline = build_pipeline(
        params, use_pos_features=False, use_relation_features=False
    )

    vectorizer = pipeline.named_steps["features"]
    assert vectorizer.method == "tfidf_bigram"
    assert vectorizer.max_features == 1234
    assert vectorizer.use_pos is False
    assert params == {"method": "tfidf_bigram", "max_features": 1234, "use_pos": False}


def test_pipeline_forwards_random_forest_hyperparameters():
    pipeline = build_pipeline(
        {"method": "count"},
        classifier_params={"n_estimators": 7, "min_samples_leaf": 3},
        use_pos_features=False,
    )
    classifier = pipeline.named_steps["classifier"]

    assert classifier.n_estimators == 7
    assert classifier.min_samples_leaf == 3


def test_pipeline_normalizes_json_style_list_params():
    pipeline = build_pipeline(
        {
            "method": "tfidf",
            "columns": ["premise_processed", "hypothesis_processed"],
            "word_ngram_range": [1, 2],
            "char_ngram_range": [3, 5],
        },
        use_pos_features=False,
        use_relation_features=False,
    )
    vectorizer = pipeline.named_steps["features"]

    assert tuple(vectorizer.columns) == ("premise_processed", "hypothesis_processed")
    assert vectorizer.word_ngram_range == (1, 2)
    assert vectorizer.char_ngram_range == (3, 5)


def test_pipeline_fit_predict_learns_synthetic_signal(synthetic_pairs):
    pipeline = build_pipeline(
        {"method": "tfidf", "min_df": 1, "max_features": 5_000},
        classifier_params={"n_estimators": 25, "min_samples_leaf": 1},
        use_pos_features=False,
        select_k=50,
    )
    pipeline.fit(synthetic_pairs, synthetic_pairs["label"])
    predictions = pipeline.predict(synthetic_pairs)

    assert predictions.shape == (len(synthetic_pairs),)
    assert set(np.unique(predictions)) <= {0, 1, 2}
    accuracy = float((predictions == synthetic_pairs["label"]).mean())
    assert accuracy > 0.8


def test_pipeline_is_reproducible_with_fixed_seed(synthetic_pairs):
    first = build_pipeline(
        {"method": "tfidf"},
        classifier_params={"n_estimators": 15},
        use_pos_features=False,
        select_k=50,
        random_state=42,
    )
    second = build_pipeline(
        {"method": "tfidf"},
        classifier_params={"n_estimators": 15},
        use_pos_features=False,
        select_k=50,
        random_state=42,
    )

    first.fit(synthetic_pairs, synthetic_pairs["label"])
    second.fit(synthetic_pairs, synthetic_pairs["label"])

    np.testing.assert_array_equal(
        first.predict(synthetic_pairs), second.predict(synthetic_pairs)
    )


def test_vectorizer_vocabulary_is_fitted_only_on_training_rows(synthetic_pairs):
    unseen = synthetic_pairs.iloc[[0]].copy()
    unseen["premise_processed"] = "uniquetokenthatonlyappearsinvalidation"
    unseen["hypothesis_processed"] = "anotheruniquetokenforvalidation"
    train = synthetic_pairs.iloc[1:].copy()

    pipeline = build_pipeline(
        {"method": "count", "min_df": 1},
        classifier_params={"n_estimators": 5},
        use_pos_features=False,
        use_relation_features=False,
        select_k=None,
    )
    pipeline.fit(train, train["label"])

    vocabulary = pipeline.named_steps["features"].vectorizer_.vocabulary_
    assert "uniquetokenthatonlyappearsinvalidation" not in vocabulary
    assert "anotheruniquetokenforvalidation" not in vocabulary

    transformed = pipeline.named_steps["features"].transform(unseen)
    assert transformed.shape[0] == 1


def test_feature_selection_is_fitted_only_on_training_rows(synthetic_pairs):
    from sklearn.feature_selection import chi2 as chi2_score

    unseen = synthetic_pairs.iloc[[0]].copy()
    unseen["premise_processed"] = "uniquetokenthatonlyappearsinvalidation"
    unseen["hypothesis_processed"] = "anotheruniquetokenforvalidation"
    train = synthetic_pairs.iloc[1:].copy()

    pipeline = build_pipeline(
        {"method": "tfidf", "min_df": 1},
        classifier_params={"n_estimators": 5},
        use_pos_features=False,
        select_k=20,
    )
    pipeline.fit(train, train["label"])

    selector = pipeline.named_steps["select"]
    assert selector.get_support().sum() == 20

    # Los scores chi² son exactamente los que produce el bloque de
    # entrenamiento: la selección nunca vio las filas reservadas.
    features = pipeline.named_steps["features"]
    train_matrix = features.transform(train)
    expected_scores, _ = chi2_score(train_matrix, train["label"])
    np.testing.assert_allclose(selector.scores_, expected_scores)

    transformed = selector.transform(features.transform(unseen))
    assert transformed.shape == (1, 20)


def test_pipeline_rejects_unknown_vectorizer_method():
    with pytest.raises(ValueError):
        build_pipeline({"method": "not-a-real-method"}, use_pos_features=False)


def test_pipeline_allows_signed_representations_without_selection():
    # El Random Forest admite componentes con signo (SVD), pero chi² exige
    # entradas no negativas: esas representaciones se usan sin selección.
    pipeline = build_pipeline(
        {"method": "tfidf_svd", "svd_components": 2},
        use_pos_features=False,
        select_k=None,
    )

    assert list(pipeline.named_steps) == ["features", "classifier"]


def test_pipeline_with_auxiliary_features_wraps_them_in_a_feature_union():
    pipeline = build_pipeline({"method": "tfidf"}, use_pos_features=True)

    union = pipeline.named_steps["features"]
    assert isinstance(union, FeatureUnion)
    assert [name for name, _ in union.transformer_list] == [
        "text",
        "pos_ratios",
        "relation",
    ]

    pos = dict(union.transformer_list)["pos_ratios"]
    assert isinstance(pos, POSRatioFeatures)
    assert pos.columns == ("premise_processed", "hypothesis_processed")

    relation = dict(union.transformer_list)["relation"]
    assert isinstance(relation, RelationFeatures)
    assert relation.columns == ("premise_processed", "hypothesis_processed")


def test_relation_features_can_be_used_without_pos_features():
    pipeline = build_pipeline(
        {"method": "tfidf"}, use_pos_features=False, use_relation_features=True
    )

    union = pipeline.named_steps["features"]
    assert [name for name, _ in union.transformer_list] == ["text", "relation"]


def test_pipeline_learns_synthetic_signal_with_pos_features(synthetic_pairs):
    with_pos = synthetic_pairs.copy()
    with_pos["premise_processed"] = with_pos["premise_processed"].map(
        lambda text: [(token, "NOUN") for token in text.split()]
    )
    with_pos["hypothesis_processed"] = with_pos["hypothesis_processed"].map(
        lambda text: [(token, "NOUN") for token in text.split()]
    )

    pipeline = build_pipeline(
        {"method": "tfidf", "min_df": 1, "use_pos": True},
        classifier_params={"n_estimators": 25, "min_samples_leaf": 1},
        use_pos_features=True,
        select_k=50,
    )

    pipeline.fit(with_pos, with_pos["label"])
    predictions = pipeline.predict(with_pos)

    assert predictions.shape == (len(with_pos),)
    accuracy = float((predictions == with_pos["label"]).mean())
    assert accuracy > 0.8


def test_pipeline_learns_synthetic_signal_with_relation_features(synthetic_pairs):
    pipeline = build_pipeline(
        {"method": "tfidf", "min_df": 1},
        classifier_params={"n_estimators": 25, "min_samples_leaf": 1},
        use_pos_features=False,
        use_relation_features=True,
        select_k=50,
    )

    pipeline.fit(synthetic_pairs, synthetic_pairs["label"])
    predictions = pipeline.predict(synthetic_pairs)

    accuracy = float((predictions == synthetic_pairs["label"]).mean())
    assert accuracy > 0.8


def test_default_configuration_uses_the_investigated_representation():
    pipeline = build_pipeline()
    vectorizer = dict(pipeline.named_steps["features"].transformer_list)["text"]

    assert vectorizer.word_ngram_range == (1, 1)
    assert vectorizer.min_df == 5
    assert vectorizer.use_pos is False
    assert pipeline.named_steps["select"].k == 300


def test_pipeline_accepts_premise_only_frames_with_single_text_methods(
    synthetic_pairs,
):
    frame = synthetic_pairs[["premise_processed", "label"]]

    pipeline = build_pipeline(
        {"method": "tfidf", "columns": ("premise_processed",), "min_df": 1},
        classifier_params={"n_estimators": 5},
        select_k=None,
    )
    pipeline.fit(frame, frame["label"])

    assert pipeline.predict(frame).shape == (len(frame),)

    union = pipeline.named_steps["features"]
    transformers = dict(union.transformer_list)
    assert [name for name, _ in union.transformer_list] == [
        "text",
        "pos_ratios",
        "relation",
    ]
    # Ratios POS de una columna (8) + señales propias de la frase (3).
    assert transformers["pos_ratios"].transform(frame).shape[1] == 8
    assert transformers["relation"].transform(frame).shape[1] == 3


def test_pair_diff_representation_requires_both_sides(synthetic_pairs):
    # ``pair_diff`` se define como |Xp - Xh|: sin hipótesis no hay diferencia
    # que calcular y el vectorizador lo rechaza de forma explícita.
    frame = synthetic_pairs[["premise_processed", "label"]]
    pipeline = build_pipeline(
        {"method": "pair_diff", "columns": ("premise_processed",), "min_df": 1},
        classifier_params={"n_estimators": 5},
        select_k=None,
    )

    with pytest.raises(ValueError, match="dos columnas"):
        pipeline.fit(frame, frame["label"])
