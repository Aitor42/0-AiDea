"""Contratos del transformador de features de relación premisa-hipótesis.

``RelationFeatures`` resume en 14 columnas las señales que el TF-IDF
normalizado pierde: negaciones, longitudes y números de cada frase y su
relación. Todo se calcula fila a fila, sin aprender del corpus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from relation_features import (
    NEGATIONS,
    RELATION_FEATURE_NAMES,
    RelationFeatures,
)


def relation_frame() -> pd.DataFrame:
    """DataFrame mínimo con pares que cubren negación, longitud y números."""
    return pd.DataFrame(
        {
            "premise_processed": [
                [("the", "DET"), ("movie", "NOUN"), ("has", "VERB"), ("3", "NUM"),
                 ("sequels", "NOUN")],
                [("the", "DET"), ("film", "NOUN"), ("was", "AUX"), ("not", "PART"),
                 ("released", "VERB"), ("in", "ADP"), ("1999", "NUM")],
            ],
            "hypothesis_processed": [
                [("the", "DET"), ("movie", "NOUN"), ("has", "VERB"), ("4", "NUM"),
                 ("sequels", "NOUN")],
                [("the", "DET"), ("film", "NOUN"), ("was", "AUX"), ("released", "VERB"),
                 ("in", "ADP"), ("1999", "NUM")],
            ],
        }
    )


def test_feature_names_are_stable_and_cover_the_three_blocks():
    assert len(RELATION_FEATURE_NAMES) == 14
    assert RELATION_FEATURE_NAMES[:4] == (
        "rel_neg_premise",
        "rel_neg_hypothesis",
        "rel_neg_abs_diff",
        "rel_neg_xor",
    )
    assert "rel_len_ratio" in RELATION_FEATURE_NAMES
    assert RELATION_FEATURE_NAMES[-1] == "rel_num_jaccard"


def test_transform_reports_negations_lengths_and_numbers():
    frame = relation_frame()

    transformer = RelationFeatures()
    output = transformer.fit_transform(frame)
    names = list(transformer.get_feature_names_out())

    assert output.shape == (2, 14)
    assert names == list(RELATION_FEATURE_NAMES)

    # Fila 0: sin negaciones; la premisa y la hipótesis tienen la misma
    # longitud y números distintos (3 frente a 4), así que el Jaccard es 0.
    assert output[0, names.index("rel_neg_premise")] == 0
    assert output[0, names.index("rel_neg_hypothesis")] == 0
    assert output[0, names.index("rel_neg_xor")] == 0
    assert output[0, names.index("rel_len_premise")] == 5
    assert output[0, names.index("rel_len_hypothesis")] == 5
    assert output[0, names.index("rel_len_abs_diff")] == 0
    assert output[0, names.index("rel_len_ratio")] == pytest.approx(5 / 6)
    assert output[0, names.index("rel_num_premise")] == 1
    assert output[0, names.index("rel_num_hypothesis")] == 1
    assert output[0, names.index("rel_num_jaccard")] == pytest.approx(0.0)

    # Fila 1: solo la premisa niega (neg_xor = 1) y comparte el número 1999.
    assert output[1, names.index("rel_neg_premise")] == 1
    assert output[1, names.index("rel_neg_hypothesis")] == 0
    assert output[1, names.index("rel_neg_abs_diff")] == 1
    assert output[1, names.index("rel_neg_xor")] == 1
    assert output[1, names.index("rel_num_jaccard")] == pytest.approx(1.0)
    assert output[1, names.index("rel_hypothesis_longer")] == 0
    assert output[1, names.index("rel_premise_longer")] == 1


def test_negations_are_detected_case_insensitively_and_via_contractions():
    frame = pd.DataFrame(
        {
            "premise_processed": [
                ["I", "do", "n't", "agree"],
                ["Nothing", "happened"],
                ["There", "is", "no", "way"],
            ],
            "hypothesis_processed": [
                ["I", "agree"],
                ["Nothing", "happened"],
                ["There", "is", "a", "way"],
            ],
        }
    )

    output = RelationFeatures().fit_transform(frame)
    names = list(RELATION_FEATURE_NAMES)

    assert output[0, names.index("rel_neg_premise")] == 1
    assert output[1, names.index("rel_neg_premise")] == 1
    assert output[1, names.index("rel_neg_xor")] == 0
    assert output[2, names.index("rel_neg_xor")] == 1


def test_plain_strings_are_tokenized_by_whitespace():
    frame = pd.DataFrame(
        {
            "premise_processed": ["the film was not released in 1999"],
            "hypothesis_processed": ["the film was released in 1999"],
        }
    )

    output = RelationFeatures().fit_transform(frame)
    names = list(RELATION_FEATURE_NAMES)

    assert output[0, names.index("rel_len_premise")] == 7
    assert output[0, names.index("rel_len_hypothesis")] == 6
    assert output[0, names.index("rel_neg_xor")] == 1
    assert output[0, names.index("rel_num_jaccard")] == pytest.approx(1.0)


def test_missing_values_yield_zero_rows():
    frame = pd.DataFrame(
        {
            "premise_processed": [None, [], float("nan")],
            "hypothesis_processed": [None, [], float("nan")],
        }
    )

    output = RelationFeatures().fit_transform(frame)

    np.testing.assert_array_equal(output, np.zeros((3, 14)))


def test_all_features_are_non_negative_for_chi2():
    output = RelationFeatures().fit_transform(relation_frame())

    assert (output >= 0).all()
    assert np.isfinite(output).all()


def test_transform_is_stateless_and_reproducible():
    frame = relation_frame()
    transformer = RelationFeatures()

    first = transformer.fit_transform(frame)
    second = transformer.transform(frame)

    np.testing.assert_array_equal(first, second)


def test_transformer_is_cloneable():
    transformer = RelationFeatures(
        columns=("premise_processed", "hypothesis_processed")
    )
    cloned = clone(transformer)

    assert cloned.get_params() == transformer.get_params()


def test_transform_requires_fit():
    with pytest.raises(NotFittedError):
        RelationFeatures().transform(relation_frame())


def test_missing_column_is_rejected_with_the_column_name():
    frame = relation_frame().drop(columns=["hypothesis_processed"])

    with pytest.raises(ValueError, match="hypothesis_processed"):
        RelationFeatures().fit(frame)


def test_single_column_emits_only_its_own_signals():
    frame = pd.DataFrame(
        {
            "premise_processed": [
                ["the", "movie", "has", "3", "sequels"],
                ["the", "film", "was", "not", "released", "in", "1999"],
            ]
        }
    )

    transformer = RelationFeatures(columns=("premise_processed",))
    output = transformer.fit_transform(frame)
    names = list(transformer.get_feature_names_out())

    assert output.shape == (2, 3)
    assert names == [
        "rel_neg_premise_processed",
        "rel_len_premise_processed",
        "rel_num_premise_processed",
    ]
    assert output[0].tolist() == [0.0, 5.0, 1.0]
    assert output[1].tolist() == [1.0, 7.0, 1.0]


def test_partial_frame_adapts_when_only_one_column_is_available():
    # El pipeline puede declarar dos columnas y recibir solo premisas: el
    # transformador emite las 3 señales de la frase disponible.
    frame = pd.DataFrame(
        {"premise_processed": ["the film was not released in 1999"]}
    )

    transformer = RelationFeatures()
    output = transformer.fit_transform(frame)

    assert output.shape == (1, 3)
    assert list(transformer.get_feature_names_out()) == [
        "rel_neg_premise_processed",
        "rel_len_premise_processed",
        "rel_num_premise_processed",
    ]
    # ``get_params`` sigue viendo las columnas declaradas originalmente.
    assert transformer.get_params()["columns"] == (
        "premise_processed",
        "hypothesis_processed",
    )


def test_more_than_two_columns_are_rejected():
    frame = relation_frame()
    frame["extra_processed"] = [["a"], ["b"]]

    with pytest.raises(ValueError, match="entre 1 y 2"):
        RelationFeatures(
            columns=("premise_processed", "hypothesis_processed", "extra_processed")
        ).fit(frame)


def test_non_dataframe_input_is_rejected():
    with pytest.raises(TypeError):
        RelationFeatures().fit(np.zeros((2, 2)))


def test_negation_vocabulary_includes_the_core_cues():
    # Las negaciones que el enunciado pide no eliminar deben estar cubiertas.
    assert {"not", "no", "never", "nothing", "cannot", "without"} <= NEGATIONS
