"""Contratos del transformador de ratios POS para el Random Forest.

El preprocesador con ``use_pos=True`` guarda las columnas ``*_processed``
como listas de tuplas ``(token, POS)`` y ``POSRatioFeatures`` las convierte
en ratios por etiqueta para premisa, hipótesis y su diferencia absoluta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from pos_features import POS_RATIO_TAGS, POSRatioFeatures


def tagged_frame() -> pd.DataFrame:
    """DataFrame mínimo con columnas ``*_processed`` de tuplas (token, POS)."""
    return pd.DataFrame(
        {
            "premise_processed": [
                [("dog", "NOUN"), ("runs", "VERB"), ("fast", "ADV")],
                [("cat", "NOUN"), ("barks", "VERB")],
            ],
            "hypothesis_processed": [
                [("cat", "NOUN"), ("sleeps", "VERB")],
                [("a", "DET"), ("dog", "NOUN")],
            ],
        }
    )


def test_default_tag_set_contains_the_informative_content_tags():
    # Conjunto usado en el cribado: etiquetas de contenido y función
    # relevantes para NLI, excluyendo las casi nulas (PUNCT, SPACE...).
    assert POS_RATIO_TAGS == ("NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "AUX")


def test_fit_transform_produces_ratios_for_both_sides_and_difference():
    df = pd.DataFrame(
        {
            "premise_processed": [[("dog", "NOUN"), ("runs", "VERB"), ("fast", "ADV")]],
            "hypothesis_processed": [[("cat", "NOUN"), ("sleeps", "VERB")]],
        }
    )

    transformer = POSRatioFeatures()
    output = transformer.fit_transform(df)

    assert output.shape == (1, 24)
    names = list(transformer.get_feature_names_out())

    premise_noun = names.index("pos_premise_processed_NOUN")
    premise_verb = names.index("pos_premise_processed_VERB")
    premise_adv = names.index("pos_premise_processed_ADV")
    hypothesis_noun = names.index("pos_hypothesis_processed_NOUN")
    hypothesis_verb = names.index("pos_hypothesis_processed_VERB")

    assert output[0, premise_noun] == pytest.approx(1 / 3)
    assert output[0, premise_verb] == pytest.approx(1 / 3)
    assert output[0, premise_adv] == pytest.approx(1 / 3)
    assert output[0, names.index("pos_premise_processed_ADJ")] == pytest.approx(0.0)
    assert output[0, hypothesis_noun] == pytest.approx(0.5)
    assert output[0, hypothesis_verb] == pytest.approx(0.5)

    diff_noun = names.index("pos_diff_NOUN")
    diff_adv = names.index("pos_diff_ADV")
    assert output[0, diff_noun] == pytest.approx(abs(1 / 3 - 0.5))
    assert output[0, diff_adv] == pytest.approx(abs(1 / 3 - 0.0))


def test_unknown_tags_count_in_the_denominator_but_not_in_the_features():
    df = pd.DataFrame(
        {
            "premise_processed": [[("x", "NOUN"), ("y", "PUNCT"), ("z", "X")]],
            "hypothesis_processed": [[("w", "NOUN")]],
        }
    )

    transformer = POSRatioFeatures()
    output = transformer.fit_transform(df)
    names = list(transformer.get_feature_names_out())

    # PUNCT no tiene columna propia, pero sí resta en el denominador.
    assert "pos_premise_processed_PUNCT" not in names
    assert output[0, names.index("pos_premise_processed_NOUN")] == pytest.approx(1 / 3)
    premise_slice = output[0, : len(POS_RATIO_TAGS)]
    assert premise_slice.sum() == pytest.approx(1 / 3)


def test_plain_strings_and_missing_values_yield_zero_rows():
    df = pd.DataFrame(
        {
            "premise_processed": ["plain lemmatized text", None, []],
            "hypothesis_processed": [[("word", "NOUN")], [("word", "NOUN")], []],
        }
    )

    output = POSRatioFeatures().fit_transform(df)
    premise_slice = output[:, : len(POS_RATIO_TAGS)]

    # Las cadenas planas (datos sin POS) y los huecos no aportan ratios.
    np.testing.assert_array_equal(premise_slice, np.zeros((3, len(POS_RATIO_TAGS))))


def test_single_column_has_no_difference_features():
    df = tagged_frame()[["premise_processed"]]

    transformer = POSRatioFeatures()
    output = transformer.fit_transform(df)

    assert output.shape == (2, 8)
    assert not any(name.startswith("pos_diff_") for name in transformer.get_feature_names_out())


def test_more_than_two_columns_are_rejected():
    df = tagged_frame()
    df["extra_processed"] = [[("a", "NOUN")], [("b", "VERB")]]

    with pytest.raises(ValueError, match="columna"):
        POSRatioFeatures(columns=("premise_processed", "hypothesis_processed", "extra_processed")).fit(df)


def test_missing_column_is_rejected_with_the_column_name():
    df = tagged_frame().drop(columns=["hypothesis_processed"])

    with pytest.raises(ValueError, match="hypothesis_processed"):
        POSRatioFeatures().fit(df)


def test_transformer_is_cloneable_and_respects_fit_transform_protocol():
    transformer = POSRatioFeatures(columns=("premise_processed",), tags=("NOUN", "VERB"))

    cloned = clone(transformer)
    assert cloned.get_params() == transformer.get_params()

    df = tagged_frame()
    fit_transformed = transformer.fit_transform(df)
    np.testing.assert_array_equal(fit_transformed, transformer.transform(df))

    with pytest.raises(NotFittedError):
        POSRatioFeatures().transform(df)


def test_custom_tags_expand_the_feature_set():
    df = pd.DataFrame(
        {
            "premise_processed": [[("x", "NOUN"), ("y", "PUNCT")]],
            "hypothesis_processed": [[("z", "VERB")]],
        }
    )

    transformer = POSRatioFeatures(tags=("NOUN", "PUNCT", "VERB"))
    output = transformer.fit_transform(df)
    names = list(transformer.get_feature_names_out())

    assert output.shape == (1, 9)  # 2 columnas × 3 etiquetas + 3 diferencias
    assert names.index("pos_premise_processed_PUNCT") >= 0
    assert output[0, names.index("pos_premise_processed_PUNCT")] == pytest.approx(0.5)
    assert output[0, names.index("pos_diff_VERB")] == pytest.approx(1.0)


def test_feature_names_are_unique_and_match_the_output_width():
    transformer = POSRatioFeatures()
    output = transformer.fit_transform(tagged_frame())
    names = list(transformer.get_feature_names_out())

    assert len(names) == output.shape[1]
    assert len(set(names)) == len(names)
